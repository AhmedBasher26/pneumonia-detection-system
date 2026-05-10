"""
app.py — Phase 5: Deployment Pipeline (Streamlit Web App)

Provides a simple UI for a user to upload a chest X-ray image,
run it through the model locally from Hugging Face Hub or via a FastAPI endpoint,
and display prediction results with Grad-CAM explainability.
"""

import os
import io
import base64
import tempfile
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras import Model  # type: ignore
from huggingface_hub import hf_hub_download

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
HF_MODEL_REPO_ID = "Ahmedb2612/pneumonia-detection-model"
HF_MODEL_FILENAME = "best_model.keras"

# Leave empty by default on Streamlit Cloud.
# If you deploy FastAPI separately, set API_URL in Streamlit secrets/environment.
API_URL = os.environ.get("API_URL", "").rstrip("/")

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
@st.cache_resource(show_spinner="Downloading/loading model from Hugging Face...")
def load_model() -> Optional[Model]:
    """Load the trained Keras model from Hugging Face Hub.

    The model file is stored outside GitHub because it is larger than GitHub's
    normal file-size limit. Streamlit downloads it once and caches it across reruns.
    """
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO_ID,
            filename=HF_MODEL_FILENAME,
        )

        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except Exception as e:
        st.error(f"Failed to load model from Hugging Face: {e}")
        return None


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
        pass  # Non-critical; do not block the UI

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

    Note:
        On Streamlit Cloud, localhost does not point to your computer.
        API_URL must be a deployed public FastAPI URL, e.g. Render/Railway.
    """
    if not API_URL:
        raise RuntimeError(
            "API_URL is not configured. Use Local Model mode, or deploy FastAPI "
            "separately and set API_URL in Streamlit Cloud."
        )

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
        help=(
            "Local Model loads the Keras model from Hugging Face. "
            "API Endpoint requires a deployed FastAPI URL."
        ),
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

    if mode == "Local Model":
        model = load_model()
        if model is not None:
            st.sidebar.success("✅ Model loaded from Hugging Face")
            st.sidebar.caption(f"Repository: `{HF_MODEL_REPO_ID}`")
            st.sidebar.caption(f"File: `{HF_MODEL_FILENAME}`")
        else:
            st.sidebar.error("❌ Model could not be loaded")
            st.sidebar.caption("Check the Hugging Face repository name and file name.")
    else:
        if API_URL:
            st.sidebar.success("✅ API URL configured")
            st.sidebar.caption(f"API URL: `{API_URL}`")
        else:
            st.sidebar.warning("⚠️ API URL is not configured")
            st.sidebar.caption("Set `API_URL` in Streamlit Cloud or use Local Model mode.")

    return {"mode": mode, "threshold": threshold}


def render_main_area(config: dict):
    """Render the main content area with upload and results."""
    st.title("🩻 Pneumonia Detection System")
    st.markdown(
        "Upload a chest X-ray image to get an AI-assisted diagnosis with "
        "explainability using Grad-CAM heatmaps."
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

    # Save uploaded file to a temporary path
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(uploaded_file, use_container_width=True)

    heatmap_fig = None

    # Run prediction
    with st.spinner("🔍 Analyzing X-ray..."):
        try:
            if config["mode"] == "Local Model":
                model = load_model()
                if model is None:
                    st.error(
                        "The model could not be loaded from Hugging Face. "
                        "Check the repository/file name or redeploy the app."
                    )
                    return

                result = predict_local(
                    model=model,
                    image_path=tmp_path,
                    confidence_threshold=config["threshold"],
                )
                heatmap_fig = result.get("heatmap_fig")

            else:
                api_result = predict_api(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                )
                result = {
                    "prediction": api_result["prediction"],
                    "confidence": api_result["confidence"],
                    "probability": api_result["probability"],
                    "is_uncertain": api_result["is_uncertain"],
                    "recommendation": api_result["recommendation"],
                }

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
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # Display results
    with col2:
        st.subheader("Diagnosis Result")

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

        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.metric("Raw Probability", f"{result['probability']:.4f}")

        if result["is_uncertain"]:
            st.warning(
                "⚡ **UNCERTAIN PREDICTION** — Confidence is below the "
                f"threshold ({config['threshold']:.0%}). "
                "This case should be reviewed by a radiologist."
            )

        st.info(f"📋 **Recommendation:** {result['recommendation']}")

    # Display Grad-CAM heatmap
    st.markdown("---")
    st.subheader("🔬 Explainability — Grad-CAM Heatmap")
    st.markdown(
        "The heatmap highlights the image regions the model focused on "
        "for its prediction. Red areas indicate high importance."
    )

    if heatmap_fig is not None:
        if isinstance(heatmap_fig, Figure):
            st.pyplot(heatmap_fig)
        elif isinstance(heatmap_fig, Image.Image):
            st.image(heatmap_fig, use_container_width=True)
    else:
        st.warning("Grad-CAM heatmap not available.")

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This system is for research and assistive purposes only. "
        "It is NOT a substitute for professional medical diagnosis. Always consult "
        "a qualified healthcare provider."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = render_sidebar()
    render_main_area(config)


if __name__ == "__main__":
    main()
