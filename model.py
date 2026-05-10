"""
model.py — Phase 2: Model Architecture & Transfer Learning

Implements the transfer learning pipeline for Pneumonia Detection with
switchable base models (ResNet50, EfficientNet-B0, MobileNetV2), a custom
classification head, and a 2-step training strategy (feature extraction →
fine-tuning).
"""

import os
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    ResNet50,
    EfficientNetB0,
    MobileNetV2,
)
from tensorflow.keras.layers import (  # type: ignore
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SHAPE = (224, 224, 3)  # Input shape: H×W×C

# Supported base model identifiers
BASE_MODELS = {
    "resnet50": ResNet50,
    "efficientnetb0": EfficientNetB0,
    "mobilenetv2": MobileNetV2,
}

# Default hyperparameters
FEATURE_EXTRACT_LR = 1e-3     # Learning rate for feature extraction step
FINE_TUNE_LR = 1e-5           # Very low LR for fine-tuning step (cite: 51)
FINE_TUNE_LAYERS = 20         # Number of top base layers to unfreeze
DROPOUT_RATE = 0.5            # Dropout before output layer
DENSE_UNITS = 128             # Dense layer units (cite: 54)
LOSS = "binary_crossentropy"  # Binary classification loss (cite: 58)


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------
def build_model(
    base_model_name: str = "resnet50",
    img_shape: tuple = IMG_SHAPE,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    trainable: bool = False,
) -> Model:
    """
    Build a transfer learning model with a pretrained base + custom head.

    Architecture (cite: 54):
        Base Model (pretrained, ImageNet) →
        GlobalAveragePooling2D →
        Dense(dense_units, ReLU) →
        Dropout(dropout_rate) →
        Dense(1, Sigmoid)

    Args:
        base_model_name: One of 'resnet50', 'efficientnetb0', 'mobilenetv2'.
        img_shape:       Input tensor shape (H, W, C).
        dense_units:     Number of units in the dense layer.
        dropout_rate:    Dropout fraction.
        trainable:       Whether the base model layers are trainable.
                         False = feature extraction mode (cite: 46-48).
                         True  = fine-tuning mode (cite: 49-51).

    Returns:
        Compiled Keras Model ready for training or inference.
    """
    base_key = base_model_name.lower().replace("-", "").replace("_", "")
    if base_key not in BASE_MODELS:
        raise ValueError(
            f"Unsupported base model: '{base_model_name}'. "
            f"Choose from: {list(BASE_MODELS.keys())}"
        )

    # --- Load pretrained base (without top classification head) ---
    base_model_fn = BASE_MODELS[base_key]
    base_model = base_model_fn(
        input_shape=img_shape,
        include_top=False,       # Remove ImageNet classifier head
        weights="imagenet",      # Pretrained on ImageNet (cite: 40-43)
    )
    base_model.trainable = trainable  # Freeze or unfreeze base (cite: 46-48)
    base_model._name = f"{base_key}_base"

    # --- Build custom classification head (cite: 54) ---
    inputs = Input(shape=img_shape, name="input_image")

    # Pass through base model; training=False keeps BatchNorm in inference
    # mode even during fine-tuning (critical for transfer learning stability)
    x = base_model(inputs, training=False)

    # Global Average Pooling: collapses (H, W, C) → (C,)
    x = GlobalAveragePooling2D(name="gap")(x)

    # Dense layer with ReLU activation
    x = Dense(dense_units, activation="relu", name="dense_head")(x)

    # Dropout for regularization
    x = Dropout(dropout_rate, name="dropout")(x)

    # Output: single sigmoid unit for binary classification
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"pneumonia_{base_key}")

    return model


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def compile_model(
    model: Model,
    learning_rate: float = FEATURE_EXTRACT_LR,
) -> Model:
    """
    Compile the model with Binary Crossentropy loss and Adam optimizer (cite: 58, 60).

    Args:
        model:         Keras Model to compile.
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        The compiled model (modified in-place, returned for convenience).
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def get_callbacks(
    model_dir: str = "models",
    patience: int = 5,
) -> list:
    """
    Return standard training callbacks.

    Callbacks:
        - EarlyStopping:      Stop if val_loss doesn't improve for `patience`
                              epochs; restore best weights.
        - ReduceLROnPlateau:  Halve LR if val_loss plateaus for 3 epochs.
        - ModelCheckpoint:    Save best model (by val_loss) to `model_dir`.

    Args:
        model_dir: Directory to save model checkpoints.
        patience:  Epochs to wait before early stopping.

    Returns:
        List of Keras callbacks.
    """
    os.makedirs(model_dir, exist_ok=True)

    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR when loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # Save best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks


# ---------------------------------------------------------------------------
# 2-Step Training Strategy
# ---------------------------------------------------------------------------
def train_feature_extraction(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 10,
    model_dir: str = "models",
) -> dict:
    """
    Step 1 — Feature Extraction (cite: 46-48).

    All pretrained base layers are FROZEN. Only the custom classification
    head (GAP → Dense → Dropout → Output) is trained. This allows the
    randomly-initialized head to learn without destroying the pretrained
    features via large gradient updates.

    Args:
        model:     Model with frozen base (built via build_model with
                   trainable=False).
        train_ds:  Training tf.data.Dataset.
        val_ds:    Validation tf.data.Dataset.
        epochs:    Max training epochs.
        model_dir: Directory for checkpoints.

    Returns:
        Training history dict.
    """
    # Ensure base is frozen
    for layer in model.layers:
        if hasattr(layer, "layers"):  # This is the base model sub-model
            layer.trainable = False

    model = compile_model(model, learning_rate=FEATURE_EXTRACT_LR)
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(model_dir, patience=5),
    )
    return history


def train_fine_tuning(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 10,
    fine_tune_layers: int = FINE_TUNE_LAYERS,
    model_dir: str = "models",
) -> dict:
    """
    Step 2 — Fine-Tuning (cite: 49-51).

    Unfreeze the last `fine_tune_layers` convolutional layers of the base
    model and train with a very low learning rate. This allows the
    pretrained features to adapt to the medical imaging domain while
    preserving the general features learned from ImageNet.

    IMPORTANT: BatchNorm layers remain in inference mode (training=False
    in the base model call) to prevent catastrophic forgetting.

    Args:
        model:            Model that has already completed feature extraction.
        train_ds:         Training tf.data.Dataset.
        val_ds:           Validation tf.data.Dataset.
        epochs:           Max training epochs for fine-tuning.
        fine_tune_layers: Number of top base layers to unfreeze.
        model_dir:        Directory for checkpoints.

    Returns:
        Training history dict.
    """
    # --- Find the base model sub-layer ---
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError("Could not locate the base model within the composite model.")

    # --- Unfreeze the base model, then re-freeze bottom layers ---
    base_model.trainable = True

    # Freeze all layers except the last `fine_tune_layers`
    total_layers = len(base_model.layers)
    freeze_until = total_layers - fine_tune_layers
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= freeze_until

    # Count trainable params for logging
    trainable_count = sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
    non_trainable_count = sum(
        tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights
    )
    print(
        f"Fine-tuning: {trainable_count:,} trainable params, "
        f"{non_trainable_count:,} non-trainable params"
    )

    # --- Recompile with very low learning rate (cite: 51) ---
    model = compile_model(model, learning_rate=FINE_TUNE_LR)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(model_dir, patience=5),
    )
    return history


# ---------------------------------------------------------------------------
# Full Training Pipeline
# ---------------------------------------------------------------------------
def train_pipeline(
    base_model_name: str = "resnet50",
    train_ds: Optional[tf.data.Dataset] = None,
    val_ds: Optional[tf.data.Dataset] = None,
    feature_epochs: int = 10,
    fine_tune_epochs: int = 10,
    model_dir: str = "models",
) -> Model:
    """
    Execute the full 2-step training pipeline.

    Step 1: Feature Extraction — freeze base, train head only.
    Step 2: Fine-Tuning — unfreeze top base layers, train with low LR.

    Args:
        base_model_name:  Which pretrained base to use.
        train_ds:         Training dataset.
        val_ds:           Validation dataset.
        feature_epochs:   Epochs for feature extraction step.
        fine_tune_epochs: Epochs for fine-tuning step.
        model_dir:        Directory for saving checkpoints.

    Returns:
        Trained Keras Model.
    """
    # --- Step 1: Feature Extraction ---
    print("=" * 60)
    print("STEP 1: FEATURE EXTRACTION (base frozen)")
    print("=" * 60)
    model = build_model(
        base_model_name=base_model_name,
        trainable=False,  # Freeze all base layers
    )
    train_feature_extraction(
        model, train_ds, val_ds,
        epochs=feature_epochs,
        model_dir=model_dir,
    )

    # --- Step 2: Fine-Tuning ---
    print("=" * 60)
    print("STEP 2: FINE-TUNING (top base layers unfrozen)")
    print("=" * 60)
    train_fine_tuning(
        model, train_ds, val_ds,
        epochs=fine_tune_epochs,
        model_dir=model_dir,
    )

    # --- Save final model ---
    final_path = os.path.join(model_dir, "final_model.keras")
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    return model


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
def load_trained_model(model_path: str) -> Model:
    """
    Load a saved Keras model from disk for inference.

    Args:
        model_path: Path to the .keras model file.

    Returns:
        Loaded Keras Model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Main entry point (smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test: build and inspect each supported base model
    for name in BASE_MODELS:
        m = build_model(base_model_name=name, trainable=False)
        m = compile_model(m)
        print(f"\n--- {name.upper()} ---")
        m.summary()
        print()
