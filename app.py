"""
app.py — Streamlit Web App for Pneumonia Detection

Cloud-ready version:
- Loads best_model.keras from Hugging Face Hub.
- Uses Local Model mode by default.
- Prevents localhost API errors on Streamlit Cloud.
- Makes Grad-CAM optional, so a Grad-CAM rendering error does not stop prediction.
"""

import os
import io
import base64
from typing import Optional, Any

import matplotlib
matplotlib.use("Agg")
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
HF_REPO_ID = "Ahmedb2612/pneumonia-detection-model"
HF_MODEL_FILENAME = "best_model.keras"
API_URL = os.environ.get("API_URL", "").strip()

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
# Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model() -> Optional[Model]:
    """Load the trained Keras model from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
        )
        return tf.keras.models.load_model(model_path, compile=False)
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
    """Run prediction using the Hugging Face-loaded Keras model."""

    # Step 1: Validate input image
    is_valid, msg = validate_image_input(image_path)
    if not is_valid:
        raise InputValidationError(f"Input validation failed: {msg}")

    # Step 2: Preprocess image
    image_tensor = preprocess_single_image(image_path)

    # Step 3: Inference
    probability = float(model.predict(image_tensor, verbose=0).flatten()[0])

    # Step 4: Confidence thresholding
    result = apply_confidence_threshold(probability, confidence_threshold)

    # Step 5: Grad-CAM is optional. Do not let Grad-CAM crash the prediction.
    heatmap_fig: Optional[Any] = None
    gradcam_error: Optional[str] = None
    try:
        heatmap = grad_cam(model, image_tensor)
        original_image = image_tensor[0].numpy()
        heatmap_fig = visualize_grad_cam(original_image, heatmap, alpha=0.4)
    except Exception as e:
        gradcam_error = str(e)

    # Step 6: Logging is optional. Do not let logging crash the UI.
    try:
        log_prediction(image_path, result, DEFAULT_DB_PATH, DEFAULT_CSV_PATH)
    except Exception:
        pass

    return {
        "prediction": result.label,
        "confidence": result.confidence,
        "probability": result.probability,
        "is_uncertain": result.is_uncertain,
        "recommendation": result.recommendation,
        "heatmap_fig": heatmap_fig,
        "gradcam_error": gradcam_error,
    }


# ---------------------------------------------------------------------------
# API Prediction Pipeline
# ---------------------------------------------------------------------------
def predict_api(image_bytes: bytes, filename: str) -> dict:
    """Run prediction by calling a deployed FastAPI /predict endpoint."""
    import requests

    if not API_URL or API_URL.rstrip("/") in {"http://localhost:8000", "http://127.0.0.1:8000"}:
        raise RuntimeError(
            "API endpoint is not configured for cloud deployment. "
            "Use Local Model mode, or set API_URL to a deployed FastAPI URL."
        )

    response = requests.post(
        f"{API_URL.rstrip('/')}/predict",
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

    mode = st.sidebar.radio(
        "Prediction Mode",
        options=["Local Model", "API Endpoint"],
        index=0,
        help="Use Local Model on Streamlit Cloud. API mode requires a deployed FastAPI URL.",
    )

    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Predictions below this confidence are flagged as uncertain.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    model = load_model()
    if model is not None:
        st.sidebar.success("✅ Model loaded from Hugging Face")
        st.sidebar.caption(f"Source: `{HF_REPO_ID}/{HF_MODEL_FILENAME}`")
    else:
        st.sidebar.error("❌ Model failed to load")
        st.sidebar.caption("Check the Hugging Face model repository and Streamlit logs.")

    return {"mode": mode, "threshold": threshold}


def render_main_area(config: dict) -> None:
    """Render the main content area with upload and results."""
    st.title("🩻 Pneumonia Detection System")
    st.markdown(
        "Upload a chest X-ray image to get an AI-assisted diagnosis. "
        "Grad-CAM is displayed when available."
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPEG, PNG, BMP, TIFF",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to begin diagnosis.")
        return

    tmp_dir = os.path.join(os.path.dirname(__file__), "logs", "uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(uploaded_file, use_container_width=True)

    with st.spinner("🔍 Analyzing X-ray..."):
        try:
            if config["mode"] == "Local Model":
                model = load_model()
                if model is None:
                    st.error("Model failed to load from Hugging Face. Check the app logs.")
                    return
                result = predict_local(model, tmp_path, config["threshold"])
                heatmap_fig = result.get("heatmap_fig")
            else:
                api_result = predict_api(uploaded_file.getvalue(), uploaded_file.name)
                result = {
                    "prediction": api_result["prediction"],
                    "confidence": api_result["confidence"],
                    "probability": api_result["probability"],
                    "is_uncertain": api_result["is_uncertain"],
                    "recommendation": api_result["recommendation"],
                    "gradcam_error": None,
                }
                heatmap_fig = None
                if api_result.get("grad_cam_heatmap"):
                    heatmap_bytes = base64.b64decode(api_result["grad_cam_heatmap"])
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
                f"threshold ({config['threshold']:.0%}). This case should be reviewed."
            )

        st.info(f"📋 **Recommendation:** {result['recommendation']}")

    st.markdown("---")
    st.subheader("🔬 Explainability — Grad-CAM Heatmap")
    st.markdown(
        "The heatmap highlights the image regions the model focused on. "
        "If unavailable, the diagnosis result above still completed successfully."
    )

    if heatmap_fig is not None:
        # Duck typing avoids isinstance errors caused by Matplotlib/PIL version differences.
        if hasattr(heatmap_fig, "savefig"):
            st.pyplot(heatmap_fig)
        else:
            st.image(heatmap_fig, use_container_width=True)
    else:
        err = result.get("gradcam_error")
        if err:
            st.warning(f"Grad-CAM heatmap not available: {err}")
        else:
            st.warning("Grad-CAM heatmap not available.")

    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This system is for research and assistive purposes only. "
        "It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare provider."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config = render_sidebar()
    render_main_area(config)


if __name__ == "__main__":
    main()
