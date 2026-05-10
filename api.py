"""
api.py — Phase 5: Deployment Pipeline (FastAPI)

Provides a robust FastAPI application with a POST /predict endpoint that
accepts a chest X-ray image and returns the prediction, confidence score,
and a Grad-CAM heatmap image encoded as base64.
"""

import io
import os
import base64
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from dataset import preprocess_single_image, IMG_SIZE
from xai import grad_cam, visualize_grad_cam
from reliability import (
    validate_image_bytes,
    InputValidationError,
    apply_confidence_threshold,
    log_prediction,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DB_PATH,
    DEFAULT_CSV_PATH,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.keras")
CONFIDENCE_THRESHOLD = float(
    os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)
)

logger = logging.getLogger("pneumonia_api")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pneumonia Detection API",
    description=(
        "Medical Image Diagnosis System for Pneumonia Detection using "
        "Chest X-rays. Returns prediction, confidence, and Grad-CAM heatmap."
    ),
    version="1.0.0",
)

# Global model reference — loaded at startup
_model: Optional[Model] = None


@app.on_event("startup")
async def load_model():
    """Load the trained Keras model at application startup."""
    global _model
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Model file not found at {MODEL_PATH}. "
            f"Endpoints will return 503 until a model is available."
        )
        _model = None
        return

    _model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded from: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Helper: Encode matplotlib figure to base64
# ---------------------------------------------------------------------------
def _fig_to_base64(fig) -> str:
    """
    Save a matplotlib Figure to a PNG buffer and encode as base64 string.

    Args:
        fig: matplotlib Figure object.

    Returns:
        Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Check if the API and model are ready."""
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_loaded": _model is not None,
    }


# ---------------------------------------------------------------------------
# Prediction Endpoint (cite: 92, 94)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)"),
):
    """
    Accept a chest X-ray image and return:
        - Prediction: "Normal" or "Pneumonia"
        - Confidence: model confidence score [0, 1]
        - Is Uncertain: whether confidence is below threshold
        - Recommendation: actionable clinical guidance
        - Grad-CAM Heatmap: base64-encoded PNG of the explainability overlay

    Steps:
        1. Read uploaded image bytes.
        2. Validate image format and integrity.
        3. Preprocess image (resize, normalize, RGB conversion).
        4. Run model inference.
        5. Apply confidence thresholding.
        6. Generate Grad-CAM heatmap.
        7. Log the prediction.
        8. Return JSON response.
    """
    # --- Check model availability ---
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a trained model is available.",
        )

    # --- Step 1: Read image bytes ---
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

    # --- Step 2: Validate input (cite: 106-108) ---
    is_valid, msg = validate_image_bytes(image_bytes, file.filename or "upload")
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Input validation failed: {msg}",
        )

    # --- Step 3: Preprocess ---
    # Save to a temp file so preprocess_single_image can read it
    tmp_path = os.path.join(
        os.path.dirname(__file__), "logs", f"tmp_{file.filename}"
    )
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    try:
        image_tensor = preprocess_single_image(tmp_path)
    except Exception as e:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=422,
            detail=f"Image preprocessing failed: {e}",
        )

    # --- Step 4: Model inference ---
    probability = float(_model.predict(image_tensor, verbose=0).flatten()[0])

    # --- Step 5: Confidence thresholding (cite: 110-112) ---
    result = apply_confidence_threshold(probability, CONFIDENCE_THRESHOLD)

    # --- Step 6: Grad-CAM heatmap (cite: 75, 79) ---
    try:
        heatmap = grad_cam(_model, image_tensor)
        # Get the original image (without batch dim) for visualization
        original_image = image_tensor[0].numpy()  # shape (224, 224, 3)
        fig = visualize_grad_cam(original_image, heatmap, alpha=0.4)
        heatmap_b64 = _fig_to_base64(fig)
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        heatmap_b64 = None

    # --- Step 7: Log the prediction (cite: 126-131) ---
    try:
        log_prediction(
            image_path=file.filename or "upload",
            result=result,
            db_path=DEFAULT_DB_PATH,
            csv_path=DEFAULT_CSV_PATH,
        )
    except Exception as e:
        logger.warning(f"Prediction logging failed: {e}")

    # --- Step 8: Clean up and return ---
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    response = {
        "prediction": result.label,
        "confidence": result.confidence,
        "probability": result.probability,
        "is_uncertain": result.is_uncertain,
        "recommendation": result.recommendation,
        "grad_cam_heatmap": heatmap_b64,
    }

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
