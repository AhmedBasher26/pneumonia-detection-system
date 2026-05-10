"""
app.py — Phase 5: Deployment Pipeline (Streamlit Web App)

Provides a simple UI for a user to upload a chest X-ray image,
run it through the model (locally or via the FastAPI endpoint),
and display the prediction results alongside the Grad-CAM visualization.
"""

import os
import io
import base64
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image


import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from dataset import preprocess_single_image
from xai import grad_cam, visualize_grad_cam
from reliability import (
    validate_image_input,
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
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Optional[Model]:
    """Load the trained Keras model (cached across Streamlit reruns)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------------------------
# Local Prediction Pipeline
# ---------------------------------------------------------------------------
def predict_local(
    model: Model,
    image_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Run prediction using the locally loaded model.

    Steps:
        1. Validate input image.
        2. Preprocess image.
        3. Run model inference.
        4. Apply confidence thresholding.
        5. Generate Grad-CAM heatmap.
        6. Log the prediction.

    Args:
        model:               Trained Keras model.
        image_path:          Path to the uploaded image.
        confidence_threshold: Minimum confidence for certain prediction.

    Returns:
        dict with prediction, confidence, heatmap figure, etc.
    """
    # Step 1: Validate
    is_valid, msg = validate_image_input(image_path)
    if not is_valid:
        raise InputValidationError(f"Input validation failed: {msg}")

    # Step 2: Preprocess
    image_tensor = preprocess_single_image(image_path)

    # Step 3: Inference
    probability = float(model.predict(image_tensor, verbose=0).flatten()[0])

    # Step 4: Confidence thresholding
    result = apply_confidence_threshold(probability, confidence_threshold)

    # Step 5: Grad-CAM
    heatmap = grad_cam(model, image_tensor)
    original_image = image_tensor[0].numpy()
    fig = visualize_grad_cam(original_image, heatmap, alpha=0.4)

    # Step 6: Log
    try:
        log_prediction(image_path, result, DEFAULT_DB_PATH, DEFAULT_CSV_PATH)
    except Exception:
        pass  # Non-critical; don't block the UI

    return {
        "prediction": result.label,
        "confidence": result.confidence,
        "probability": result.probability,
        "is_uncertain": result.is_uncertain,
        "recommendation": result.recommendation,
        "heatmap_fig": fig,
    }


# ---------------------------------------------------------------------------
# API Prediction Pipeline (calls FastAPI endpoint)
# ---------------------------------------------------------------------------
def predict_api(image_bytes: bytes, filename: str) -> dict:
    """
    Run prediction by calling the FastAPI /predict endpoint.

    Args:
        image_bytes: Raw image bytes.
        filename:    Original filename.

    Returns:
        dict with prediction, confidence, heatmap base64, etc.
    """
    import requests

    response = requests.post(
        f"{API_URL}/predict",
        files={"file": (filename, image_bytes, "image/jpeg")},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    """Render the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # Prediction mode
    mode = st.sidebar.radio(
        "Prediction Mode",
        options=["Local Model", "API Endpoint"],
        index=0,
        help="Choose whether to use a locally loaded model or call the FastAPI endpoint.",
    )

    # Confidence threshold slider
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Predictions below this confidence are flagged as 'Uncertain'.",
    )

    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    model = load_model()
    if model is not None:
        st.sidebar.success("✅ Model loaded")
        st.sidebar.caption(f"Path: `{MODEL_PATH}`")
    else:
        st.sidebar.error("❌ No model found")
        st.sidebar.caption(f"Expected at: `{MODEL_PATH}`")

    return {"mode": mode, "threshold": threshold}


def render_main_area(config: dict):
    """Render the main content area with upload and results."""
    st.title("🩻 Pneumonia Detection System")
    st.markdown(
        "Upload a chest X-ray image to get an AI-assisted diagnosis with "
        "explainability (Grad-CAM heatmap)."
    )
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPEG, PNG, BMP, TIFF",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to begin diagnosis.")
        return

    # --- Save uploaded file to a temp path ---
    tmp_dir = os.path.join(os.path.dirname(__file__), "logs", "uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- Display uploaded image ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(uploaded_file, use_container_width=True)

    # --- Run prediction ---
    with st.spinner("🔍 Analyzing X-ray..."):
        try:
            if config["mode"] == "Local Model":
                model = load_model()
                if model is None:
                    st.error(
                        "No trained model found. Please train a model first "
                        "or switch to API mode."
                    )
                    return
                result = predict_local(
                    model, tmp_path, config["threshold"]
                )
                # Extract heatmap figure from local prediction
                heatmap_fig = result.get("heatmap_fig")
            else:
                # API mode
                api_result = predict_api(
                    uploaded_file.getvalue(), uploaded_file.name
                )
                result = {
                    "prediction": api_result["prediction"],
                    "confidence": api_result["confidence"],
                    "probability": api_result["probability"],
                    "is_uncertain": api_result["is_uncertain"],
                    "recommendation": api_result["recommendation"],
                }
                # Decode base64 heatmap from API
                heatmap_fig = None
                if api_result.get("grad_cam_heatmap"):
                    heatmap_b64 = api_result["grad_cam_heatmap"]
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_fig = Image.open(io.BytesIO(heatmap_bytes))

        except InputValidationError as e:
            st.error(f"❌ Input validation failed: {e}")
            return
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

    # --- Display results ---
    with col2:
        st.subheader("Diagnosis Result")

        # Prediction label with color coding
        if result["prediction"] == "Pneumonia":
            if result["is_uncertain"]:
                st.warning("⚠️ **PNEUMONIA** (Uncertain)")
            else:
                st.error("🔴 **PNEUMONIA DETECTED**")
        else:
            if result["is_uncertain"]:
                st.warning("⚠️ **NORMAL** (Uncertain)")
            else:
                st.success("🟢 **NORMAL**")

        # Metrics
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.metric("Raw Probability", f"{result['probability']:.4f}")

        # Uncertainty flag
        if result["is_uncertain"]:
            st.warning(
                "⚡ **UNCERTAIN PREDICTION** — Confidence is below the "
                f"threshold ({config['threshold']:.0%}). "
                "This case should be reviewed by a radiologist."
            )

        # Recommendation
        st.info(f"📋 **Recommendation:** {result['recommendation']}")

    # --- Display Grad-CAM heatmap ---
    st.markdown("---")
    st.subheader("🔬 Explainability — Grad-CAM Heatmap")
    st.markdown(
        "The heatmap highlights the image regions the model focused on "
        "for its prediction. Red areas indicate high importance."
    )

    if heatmap_fig is not None:
        if isinstance(heatmap_fig, plt.Figure):
            st.pyplot(heatmap_fig)
        elif isinstance(heatmap_fig, Image.Image):
            st.image(heatmap_fig, use_container_width=True)
    else:
        st.warning("Grad-CAM heatmap not available.")

    # --- Disclaimer ---
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This system is for research and assistive "
        "purposes only. It is NOT a substitute for professional medical "
        "diagnosis. Always consult a qualified healthcare provider."
    )

    # --- Clean up temp file ---
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = render_sidebar()
    render_main_area(config)


if __name__ == "__main__":
    main()
