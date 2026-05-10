"""
dataset.py — Phase 1: Data Preparation & Preprocessing

Handles dataset loading, preprocessing, and augmentation for the Pneumonia
Detection system. Assumes a directory structure of:
    data/
    ├── train/
    │   ├── normal/
    │   └── pneumonia/
    ├── val/
    │   ├── normal/
    │   └── pneumonia/
    └── test/
        ├── normal/
        └── pneumonia/
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    RandomRotation,
    RandomFlip,
    RandomZoom,
    RandomBrightness,
)
from tensorflow.keras.models import Sequential  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)       # Target resize dimensions (H, W)
BATCH_SIZE = 32             # Mini-batch size
AUTOTUNE = tf.data.AUTOTUNE # tf.data prefetch parallelism
COLOR_MODE = "rgb"          # All images converted to 3-channel RGB
LABEL_MODE = "binary"       # Binary classification: 0=normal, 1=pneumonia


# ---------------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------------
def build_augmentation_pipeline() -> Sequential:
    """
    Build a Keras Sequential model that applies on-the-fly data augmentation.

    Augmentations applied (cite: 34-37):
        - RandomRotation:   ±20° rotation
        - RandomFlip:       Horizontal flip (mirror — anatomically reasonable
                            for chest X-rays when combined with other transforms)
        - RandomZoom:       ±20% zoom in/out
        - RandomBrightness: ±20% brightness shift

    Returns:
        Sequential: A Keras model that applies augmentations when called.
    """
    augmentation = Sequential(
        [
            RandomRotation(factor=0.2, name="random_rotation"),       # ±20°
            RandomFlip(mode="horizontal", name="random_flip"),        # Horizontal
            RandomZoom(height_factor=0.2, name="random_zoom"),       # ±20%
            RandomBrightness(factor=0.2, name="random_brightness"),  # ±20%
        ],
        name="data_augmentation",
    )
    return augmentation


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------
def load_dataset(
    data_dir: str,
    subset: str = "train",
    batch_size: int = BATCH_SIZE,
    img_size: tuple = IMG_SIZE,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Load an image dataset from a directory and return a preprocessed
    tf.data.Dataset pipeline.

    Args:
        data_dir:   Path to the root data directory (e.g., './data').
        subset:     One of 'train', 'val', 'test'. Used to select the
                    correct subdirectory.
        batch_size: Number of images per batch.
        img_size:   Target (height, width) to resize images to.
        augment:    Whether to apply data augmentation (should be True
                    only for the training split).

    Returns:
        tf.data.Dataset yielding (image_batch, label_batch) where images
        are float32 in [0, 1] with shape (H, W, 3) and labels are float32
        scalars (0.0 or 1.0).
    """
    subset_dir = os.path.join(data_dir, subset)
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(
            f"Subset directory not found: {subset_dir}. "
            f"Expected structure: {data_dir}/{subset}/normal/ and "
            f"{data_dir}/{subset}/pneumonia/"
        )

    # --- Step 1: Load raw images from directory ---
    # image_dataset_from_directory infers labels from subdirectory names
    # (alphabetical: "normal"=0, "pneumonia"=1)
    dataset = image_dataset_from_directory(
        directory=subset_dir,
        labels="inferred",
        label_mode=LABEL_MODE,       # Binary: 0 or 1
        color_mode=COLOR_MODE,       # Convert grayscale → RGB
        image_size=img_size,         # Resize to 224×224
        batch_size=batch_size,
        shuffle=(subset == "train"), # Shuffle only training data
        seed=42 if subset == "train" else None,
    )

    # --- Step 2: Normalize pixel values to [0, 1] (cite: 31) ---
    # Keras image_dataset_from_directory loads uint8 [0,255].
    # We rescale to float32 [0,1] for neural network input.
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    dataset = dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE,
    )

    # --- Step 3: Apply augmentation (training only) (cite: 34-37) ---
    if augment:
        augmentation_pipeline = build_augmentation_pipeline()
        dataset = dataset.map(
            lambda x, y: (augmentation_pipeline(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # --- Step 4: Performance optimization ---
    # Prefetch overlaps data preprocessing with model execution
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


# ---------------------------------------------------------------------------
# Convenience: Load all splits
# ---------------------------------------------------------------------------
def load_all_splits(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    img_size: tuple = IMG_SIZE,
) -> dict:
    """
    Load train, validation, and test splits in one call.

    Args:
        data_dir:   Root data directory.
        batch_size: Batch size for all splits.
        img_size:   Target image size.

    Returns:
        dict with keys 'train', 'val', 'test', each mapping to a
        tf.data.Dataset. Augmentation is applied only to 'train'.
    """
    train_ds = load_dataset(
        data_dir, subset="train", batch_size=batch_size,
        img_size=img_size, augment=True,
    )
    val_ds = load_dataset(
        data_dir, subset="val", batch_size=batch_size,
        img_size=img_size, augment=False,
    )
    test_ds = load_dataset(
        data_dir, subset="test", batch_size=batch_size,
        img_size=img_size, augment=False,
    )

    return {"train": train_ds, "val": val_ds, "test": test_ds}


# ---------------------------------------------------------------------------
# Single-image preprocessing utility (for inference)
# ---------------------------------------------------------------------------
def preprocess_single_image(image_path: str, img_size: tuple = IMG_SIZE) -> tf.Tensor:
    """
    Load and preprocess a single image file for model inference.

    Steps:
        1. Read the file from disk.
        2. Decode the image (auto-detect format: JPEG, PNG, etc.).
        3. Convert grayscale to 3-channel RGB if needed.
        4. Resize to target dimensions.
        5. Rescale pixel values to [0, 1].
        6. Add batch dimension → shape (1, H, W, 3).

    Args:
        image_path: Absolute or relative path to the image file.
        img_size:   Target (height, width).

    Returns:
        tf.Tensor of shape (1, H, W, 3) with float32 values in [0, 1].
    """
    # Read raw bytes
    raw = tf.io.read_file(image_path)

    # Decode — channels=3 forces RGB output even for grayscale input (cite: 32)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)          # Resize to 224×224
    img = tf.cast(img, tf.float32) / 255.0         # Normalize to [0, 1]
    img = tf.expand_dims(img, axis=0)              # Add batch dim: (1,224,224,3)

    return img


# ---------------------------------------------------------------------------
# Main entry point (quick smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    print(f"Loading datasets from: {DATA_DIR}")
    splits = load_all_splits(DATA_DIR)

    for split_name, ds in splits.items():
        # Take one batch to verify shape
        for images, labels in ds.take(1):
            print(
                f"[{split_name}] batch shape: images={images.shape}, "
                f"labels={labels.shape}, dtype={images.dtype}"
            )
            print(
                f"  Pixel range: [{images.numpy().min():.3f}, "
                f"{images.numpy().max():.3f}]"
            )
