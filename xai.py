"""
xai.py — Phase 3: Evaluation & Explainability (XAI)

Provides model evaluation metrics (Accuracy, Precision, Recall, F1),
confusion matrix visualization, and Grad-CAM heatmap generation for
explainability — critical for building trust in medical AI systems.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Pneumonia"]
THRESHOLD = 0.5  # Binary classification threshold


# ---------------------------------------------------------------------------
# Prediction Helper
# ---------------------------------------------------------------------------
def get_predictions(
    model: Model,
    dataset: tf.data.Dataset,
    threshold: float = THRESHOLD,
) -> tuple:
    """
    Run inference on a dataset and return true labels + predicted labels +
    raw probability scores.

    Args:
        model:     Trained Keras model.
        dataset:   tf.data.Dataset yielding (images, labels) batches.
        threshold: Classification threshold for positive class.

    Returns:
        Tuple of (y_true, y_pred, y_prob):
            y_true: np.ndarray of ground-truth labels (0 or 1).
            y_pred: np.ndarray of predicted labels (0 or 1).
            y_prob: np.ndarray of raw sigmoid probabilities.
    """
    y_true_list = []
    y_prob_list = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0).flatten()
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy().flatten())

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = (y_prob >= threshold).astype(int)

    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Metrics Computation (cite: 62-65)
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute classification metrics.

    Emphasis on Recall (cite: 71): In medical diagnosis, false negatives
    (missing a pneumonia case) are far more dangerous than false positives
    (flagging a healthy patient for review). Maximizing Recall minimizes
    the chance of missing a real pneumonia case.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        dict with keys: accuracy, precision, recall, f1_score.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 50)
    print("         MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}  <- CRITICAL (minimize FN)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Confusion Matrix Visualization (cite: 72)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot a confusion matrix for binary classification.

    Layout:
        - Rows:    True labels (Normal / Pneumonia)
        - Columns: Predicted labels (Normal / Pneumonia)

    Args:
        y_true:    Ground-truth labels.
        y_pred:    Predicted labels.
        save_path: Optional path to save the figure (PNG).

    Returns:
        matplotlib Figure object.
    """
    cm_matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_matrix,
        display_labels=CLASS_NAMES,
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Pneumonia Detection")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Grad-CAM Implementation (cite: 75, 79, 81, 83)
# ---------------------------------------------------------------------------
def _find_last_conv_layer(model: Model) -> str:
    """
    Find the name of the last convolutional layer in the model.

    Grad-CAM requires gradients with respect to a convolutional feature map.
    We use the last conv layer to get the most spatially-detailed heatmap.

    Args:
        model: Keras Model.

    Returns:
        Name of the last convolutional layer.
    """
    last_conv_name = None
    for layer in reversed(model.layers):
        # Check if this layer is a Conv2D directly
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
            break
        # If it's a nested model (e.g., the base model), search inside it
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_name = sub_layer.name
                    break
            if last_conv_name:
                break

    if last_conv_name is None:
        raise ValueError(
            "No Conv2D layer found in the model. "
            "Grad-CAM requires at least one convolutional layer."
        )
    return last_conv_name


def _build_grad_cam_model(
    model: Model,
    last_conv_layer_name: str,
) -> Model:
    """
    Build a sub-model that outputs both the last conv layer's feature map
    and the final predictions. Handles nested sub-models (e.g., base model
    inside the composite model).

    Args:
        model:                 The full composite model.
        last_conv_layer_name:  Name of the target conv layer.

    Returns:
        Keras Model with two outputs: [conv_output, prediction].
    """
    # Try to find the layer directly in the top-level model
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
        return grad_model
    except ValueError:
        pass

    # If not found at top level, it's inside a sub-model (e.g., the base).
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            try:
                conv_layer = layer.get_layer(last_conv_layer_name)
                # Build a sub-model for the base that outputs both
                # the conv feature map and the base's final output
                sub_grad_model = Model(
                    inputs=layer.input,
                    outputs=[conv_layer.output, layer.output],
                )
                # Now build the full grad model:
                # Input -> sub_grad_model -> [conv_out, base_out]
                # base_out -> rest of composite model -> final prediction
                inputs = model.input
                conv_out, base_out = sub_grad_model(inputs)

                # Pass base_out through all layers after the base model
                x = base_out
                found_base = False
                for comp_layer in model.layers:
                    if comp_layer == layer:
                        found_base = True
                        continue
                    if found_base and not isinstance(comp_layer, tf.keras.layers.InputLayer):
                        x = comp_layer(x)

                grad_model = Model(
                    inputs=inputs,
                    outputs=[conv_out, x],
                )
                return grad_model
            except ValueError:
                continue

    raise ValueError(
        f"Could not find conv layer '{last_conv_layer_name}' "
        f"in the model or any sub-model."
    )


def grad_cam(
    model: Model,
    image: np.ndarray,
    last_conv_layer_name: str = None,
    pred_index: int = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single input image (cite: 75, 79).

    Grad-CAM (Gradient-weighted Class Activation Mapping) produces a coarse
    heatmap highlighting the image regions most influential for the model's
    prediction. This is essential for:
        - Building clinician trust in the model's reasoning (cite: 81).
        - Detecting when the model focuses on irrelevant regions (cite: 83).

    Algorithm:
        1. Forward pass the image through the model to get predictions.
        2. Compute gradients of the target class score w.r.t. the last
           conv layer's output feature map.
        3. Pool the gradients over spatial dimensions (global average)
           to get per-channel importance weights.
        4. Compute a weighted sum of feature maps -> raw heatmap.
        5. ReLU the heatmap (keep only positive contributions) and
           normalize to [0, 1].

    Args:
        model:               Trained Keras Model.
        image:               Preprocessed input image, shape (1, H, W, 3),
                             float32 in [0, 1].
        last_conv_layer_name: Name of the last conv layer. If None, auto-detect.
        pred_index:          Class index for gradient computation.
                             None = use the top predicted class.

    Returns:
        Heatmap as np.ndarray of shape (H, W) with values in [0, 1].
    """
    # --- Auto-detect last conv layer if not specified ---
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    # --- Build a model that outputs both conv features and predictions ---
    grad_model = _build_grad_cam_model(model, last_conv_layer_name)

    # --- Compute gradients ---
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        # Score for the target class
        class_channel = predictions[:, pred_index]

    # Gradient of the target class score w.r.t. the conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Pool gradients over spatial dims -> per-channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]  # Remove batch dim
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: keep only positive contributions and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap


# ---------------------------------------------------------------------------
# Grad-CAM Visualization
# ---------------------------------------------------------------------------
def visualize_grad_cam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: str = None,
) -> plt.Figure:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image:     Original image array, shape (H, W, 3), float32 [0,1].
        heatmap:   Grad-CAM heatmap, shape (h, w), values [0,1].
        alpha:     Transparency of the heatmap overlay (0=only image, 1=only heatmap).
        save_path: Optional path to save the figure (PNG).

    Returns:
        matplotlib Figure with the overlaid visualization.
    """
    # Resize heatmap to match image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis],  # Add channel dim for resize
        size=(image.shape[0], image.shape[1]),
    ).numpy().squeeze()

    # Apply a color map (jet) to the heatmap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel

    # Superimpose: weighted sum of original image and heatmap
    display_image = np.clip(image, 0, 1)
    superimposed = display_image * (1 - alpha) + heatmap_colored * alpha
    superimposed = np.clip(superimposed, 0, 1)

    # --- Plot side-by-side ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(display_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay (X-ray + Heatmap)")
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Explainability Visualization", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM visualization saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Model,
    test_ds: tf.data.Dataset,
    save_dir: str = "logs",
) -> dict:
    """
    Run the full evaluation pipeline on the test set.

    Steps:
        1. Get predictions on the test dataset.
        2. Compute metrics (Accuracy, Precision, Recall, F1).
        3. Print metrics.
        4. Plot and save the confusion matrix.

    Args:
        model:    Trained Keras Model.
        test_ds:  Test tf.data.Dataset.
        save_dir: Directory to save evaluation artifacts.

    Returns:
        dict with 'metrics' and 'predictions' keys.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Get predictions
    y_true, y_pred, y_prob = get_predictions(model, test_ds)

    # Step 2: Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # Step 3: Print metrics
    print_metrics(metrics)

    # Step 4: Plot confusion matrix
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)

    return {
        "metrics": metrics,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        },
    }


# ---------------------------------------------------------------------------
# Main entry point (smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("xai.py loaded. Use evaluate_model() or grad_cam() from other modules.")
