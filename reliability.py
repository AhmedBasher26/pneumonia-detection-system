"""
reliability.py — Phase 4: System Reliability Enhancements

Implements input validation, confidence thresholding with "Uncertain"
flagging, and a logging system (SQLite + CSV fallback) to ensure the
pneumonia detection system is safe, auditable, and production-ready.
"""

import os
import csv
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Allowed image formats for medical X-ray input (cite: 106-108)
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "GIF"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Minimum image dimensions (pixels) — reject tiny/non-medical inputs
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Confidence threshold for the "Uncertain" flag (cite: 110-112, 123-124)
# If the model's confidence is below this value, the prediction is flagged
# as "Uncertain" and a human-in-the-loop (refer to doctor) is triggered.
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# Logging database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "logs", "predictions.db"
)
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "logs", "predictions.csv"
)

# Module-level logger
logger = logging.getLogger("pneumonia_reliability")
logger.setLevel(logging.INFO)


# ===========================================================================
# Input Validation (cite: 106-108)
# ===========================================================================
class InputValidationError(Exception):
    """Raised when an input image fails validation checks."""
    pass


def validate_image_input(image_path: str) -> Tuple[bool, str]:
    """
    Validate that the input file is a legitimate medical image suitable
    for the pneumonia detection model.

    Checks performed (cite: 106-108):
        1. File exists on disk.
        2. File extension is in the allowed set.
        3. File can be opened as an image (PIL).
        4. Image format is in the allowed set (JPEG, PNG, etc.).
        5. Image dimensions meet minimum size requirements (reject
           tiny or corrupted inputs that are unlikely to be real X-rays).

    Args:
        image_path: Path to the image file to validate.

    Returns:
        Tuple of (is_valid: bool, message: str).
        If is_valid is True, message contains "Valid".
        If is_valid is False, message contains the reason for rejection.
    """
    # --- Check 1: File exists ---
    if not os.path.isfile(image_path):
        return False, f"File not found: {image_path}"

    # --- Check 2: File extension ---
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Invalid file extension: '{ext}'. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # --- Check 3: Can be opened as an image ---
    try:
        img = Image.open(image_path)
        img.verify()  # Verify integrity without fully decoding
        img = Image.open(image_path)  # Re-open for further checks
    except Exception as e:
        return False, f"Cannot open image: {e}"

    # --- Check 4: Image format ---
    if img.format not in ALLOWED_FORMATS:
        return False, (
            f"Invalid image format: '{img.format}'. "
            f"Allowed: {sorted(ALLOWED_FORMATS)}"
        )

    # --- Check 5: Minimum dimensions ---
    width, height = img.size
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return False, (
            f"Image too small: {width}x{height}px. "
            f"Minimum: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px"
        )

    return True, "Valid"


def validate_image_bytes(image_bytes: bytes, filename: str = "upload") -> Tuple[bool, str]:
    """
    Validate an image provided as raw bytes (e.g., from an API upload).

    Args:
        image_bytes: Raw image bytes.
        filename:    Original filename (used for extension check).

    Returns:
        Tuple of (is_valid: bool, message: str).
    """
    # --- Check extension from filename ---
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Invalid file extension: '{ext}'. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # --- Check: Can be decoded ---
    try:
        import io
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return False, f"Cannot decode image bytes: {e}"

    # --- Check format ---
    if img.format not in ALLOWED_FORMATS:
        return False, f"Invalid image format: '{img.format}'"

    # --- Check dimensions ---
    width, height = img.size
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return False, (
            f"Image too small: {width}x{height}px. "
            f"Minimum: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px"
        )

    return True, "Valid"


# ===========================================================================
# Confidence Thresholding (cite: 110-112, 123-124)
# ===========================================================================
class PredictionResult:
    """
    Encapsulates a model prediction with confidence-based reliability flagging.

    Attributes:
        label:          "Normal" or "Pneumonia" based on threshold 0.5.
        confidence:     Raw sigmoid probability (float in [0, 1]).
        is_uncertain:   True if confidence < confidence_threshold (cite: 110-112).
        recommendation: Actionable recommendation string.
    """

    def __init__(
        self,
        probability: float,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Args:
            probability:           Raw sigmoid output from the model.
            confidence_threshold:  Minimum confidence for a certain prediction.
        """
        self.probability = float(probability)
        self.confidence_threshold = confidence_threshold

        # Binary label: sigmoid >= 0.5 → Pneumonia, else Normal
        self.label = "Pneumonia" if self.probability >= 0.5 else "Normal"

        # Confidence = max(prob, 1-prob) — how sure the model is
        self.confidence = max(self.probability, 1.0 - self.probability)

        # Uncertain flag (cite: 110-112, 123-124)
        self.is_uncertain = self.confidence < self.confidence_threshold

        # Recommendation
        if self.is_uncertain:
            self.recommendation = (
                "UNCERTAIN — Confidence below threshold. "
                "Refer to a radiologist for manual review."
            )
        else:
            self.recommendation = (
                f"Confident prediction: {self.label}. "
                f"Standard clinical workflow applies."
            )

    def to_dict(self) -> dict:
        """Serialize the prediction result to a dictionary."""
        return {
            "label": self.label,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
            "is_uncertain": self.is_uncertain,
            "recommendation": self.recommendation,
        }

    def __repr__(self) -> str:
        flag = " [UNCERTAIN]" if self.is_uncertain else ""
        return (
            f"PredictionResult(label={self.label}, "
            f"confidence={self.confidence:.2%}{flag})"
        )


def apply_confidence_threshold(
    probability: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> PredictionResult:
    """
    Apply confidence thresholding to a model's raw sigmoid output.

    If confidence < threshold, the result is flagged as "Uncertain" to
    trigger a human-in-the-loop review (cite: 110-112, 123-124).

    Args:
        probability: Raw sigmoid probability from the model.
        threshold:   Minimum confidence for a certain prediction.

    Returns:
        PredictionResult with label, confidence, and uncertainty flag.
    """
    return PredictionResult(probability, confidence_threshold=threshold)


# ===========================================================================
# Logging System (cite: 126-131)
# ===========================================================================
def _init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Initialize the SQLite database for prediction logging.

    Creates the 'predictions' table if it doesn't exist.

    Schema (cite: 126, 128-131):
        - id:            Auto-incrementing primary key.
        - timestamp:     ISO 8601 UTC timestamp.
        - image_path:    Path or identifier for the input image.
        - prediction:    "Normal" or "Pneumonia".
        - confidence:    Model confidence score [0, 1].
        - is_uncertain:  Whether the prediction was flagged uncertain.
        - recommendation: Actionable recommendation string.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection to the database.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            image_path    TEXT    NOT NULL,
            prediction    TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            is_uncertain  INTEGER NOT NULL DEFAULT 0,
            recommendation TEXT
        )
    """)
    conn.commit()
    return conn


def log_prediction_sqlite(
    image_path: str,
    result: PredictionResult,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Log a prediction to the SQLite database (cite: 126, 128-131).

    Args:
        image_path: Path or identifier for the input image.
        result:     PredictionResult from apply_confidence_threshold().
        db_path:    Path to the SQLite database file.
    """
    conn = _init_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            """INSERT INTO predictions
               (timestamp, image_path, prediction, confidence,
                is_uncertain, recommendation)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                image_path,
                result.label,
                result.confidence,
                int(result.is_uncertain),
                result.recommendation,
            ),
        )
        conn.commit()
        logger.info(
            f"Logged prediction: {result.label} "
            f"(confidence={result.confidence:.2%}) for {image_path}"
        )
    finally:
        conn.close()


def log_prediction_csv(
    image_path: str,
    result: PredictionResult,
    csv_path: str = DEFAULT_CSV_PATH,
) -> None:
    """
    Log a prediction to a CSV file as a fallback or alternative to SQLite.

    CSV columns: timestamp, image_path, prediction, confidence,
                 is_uncertain, recommendation

    Args:
        image_path: Path or identifier for the input image.
        result:     PredictionResult from apply_confidence_threshold().
        csv_path:   Path to the CSV log file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header only if file is new
        if not file_exists:
            writer.writerow([
                "timestamp", "image_path", "prediction",
                "confidence", "is_uncertain", "recommendation",
            ])
        writer.writerow([
            timestamp,
            image_path,
            result.label,
            f"{result.confidence:.4f}",
            result.is_uncertain,
            result.recommendation,
        ])

    logger.info(f"Logged prediction to CSV: {image_path}")


def log_prediction(
    image_path: str,
    result: PredictionResult,
    db_path: str = DEFAULT_DB_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
    use_sqlite: bool = True,
) -> None:
    """
    Log a prediction using the configured backend (SQLite by default,
    CSV as fallback).

    If SQLite fails, automatically falls back to CSV logging.

    Args:
        image_path:  Path or identifier for the input image.
        result:      PredictionResult from apply_confidence_threshold().
        db_path:     Path to the SQLite database.
        csv_path:    Path to the CSV log file.
        use_sqlite:  If True, try SQLite first; always use CSV as fallback.
    """
    if use_sqlite:
        try:
            log_prediction_sqlite(image_path, result, db_path)
            return
        except Exception as e:
            logger.warning(
                f"SQLite logging failed ({e}), falling back to CSV."
            )
    # Fallback to CSV
    log_prediction_csv(image_path, result, csv_path)


def query_logs(
    limit: int = 50,
    db_path: str = DEFAULT_DB_PATH,
) -> list:
    """
    Query recent prediction logs from the SQLite database.

    Args:
        limit:   Maximum number of records to return.
        db_path: Path to the SQLite database.

    Returns:
        List of dicts with prediction log records.
    """
    conn = _init_db(db_path)
    try:
        cursor = conn.execute(
            """SELECT id, timestamp, image_path, prediction,
                      confidence, is_uncertain, recommendation
               FROM predictions
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()


# ===========================================================================
# Full Reliability Pipeline
# ===========================================================================
def reliable_predict(
    model,
    image_tensor: np.ndarray,
    image_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    db_path: str = DEFAULT_DB_PATH,
    csv_path: str = DEFAULT_CSV_PATH,
) -> dict:
    """
    Run a prediction through the full reliability pipeline:

        1. Validate input image (cite: 106-108).
        2. Run model inference.
        3. Apply confidence threshold (cite: 110-112).
        4. Log the prediction (cite: 126-131).

    Args:
        model:               Trained Keras model.
        image_tensor:        Preprocessed image tensor, shape (1,224,224,3).
        image_path:          Original image path (for logging).
        confidence_threshold: Minimum confidence for certain prediction.
        db_path:             SQLite database path.
        csv_path:            CSV log path.

    Returns:
        dict with all prediction details including uncertainty flag.
    """
    # Step 1: Validate input
    is_valid, msg = validate_image_input(image_path)
    if not is_valid:
        raise InputValidationError(f"Input validation failed: {msg}")

    # Step 2: Model inference
    probability = float(model.predict(image_tensor, verbose=0).flatten()[0])

    # Step 3: Apply confidence threshold
    result = apply_confidence_threshold(probability, confidence_threshold)

    # Step 4: Log the prediction
    log_prediction(image_path, result, db_path, csv_path)

    return result.to_dict()


# ---------------------------------------------------------------------------
# Main entry point (smoke test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test confidence thresholding
    for prob in [0.99, 0.87, 0.72, 0.50, 0.30, 0.05]:
        result = apply_confidence_threshold(prob)
        print(result)

    # Test logging (CSV fallback — no model needed)
    result = apply_confidence_threshold(0.92)
    log_prediction("test_image.jpeg", result, use_sqlite=False)
    print("CSV log test completed.") 
