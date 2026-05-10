# 🩻 Pneumonia Detection System — Chest X-ray AI Diagnosis

A complete, production-ready medical image diagnosis system for **Pneumonia Detection** using chest X-rays, built with transfer learning, explainability (Grad-CAM), reliability safeguards, and a full deployment pipeline.

---

## 📁 Project Structure

```
pneumonia-detection-system/
├── data/                        # Dataset (not included — see below)
│   ├── train/
│   │   ├── normal/
│   │   └── pneumonia/
│   ├── val/
│   │   ├── normal/
│   │   └── pneumonia/
│   └── test/
│       ├── normal/
│       └── pneumonia/
├── models/                      # Saved model checkpoints
├── logs/                        # Prediction logs (SQLite + CSV)
├── dataset.py                   # Phase 1: Data prep & preprocessing
├── model.py                     # Phase 2: Architecture & transfer learning
├── xai.py                       # Phase 3: Evaluation & Grad-CAM
├── reliability.py               # Phase 4: Input validation, confidence, logging
├── api.py                       # Phase 5: FastAPI endpoint
├── app.py                       # Phase 5: Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your chest X-ray dataset in the `data/` directory following the structure above. The [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset is recommended.

### 3. Train the Model

```python
from dataset import load_all_splits
from model import train_pipeline

splits = load_all_splits("./data")
model = train_pipeline(
    base_model_name="resnet50",  # or "efficientnetb0" or "mobilenetv2"
    train_ds=splits["train"],
    val_ds=splits["val"],
    feature_epochs=10,
    fine_tune_epochs=10,
)
```

### 4. Evaluate

```python
from xai import evaluate_model

results = evaluate_model(model, splits["test"], save_dir="logs")
```

### 5. Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test with curl:
```bash
curl -X POST -F "file=@xray.jpg" http://localhost:8000/predict
```

### 6. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🏗️ Architecture Overview

| Phase | Module | Key Features |
|-------|--------|---------------|
| 1 — Data Prep | `dataset.py` | Resize 224×224, normalize [0,1], grayscale→RGB, augmentation (rotation, flip, zoom, brightness) |
| 2 — Model | `model.py` | ResNet50 / EfficientNet-B0 / MobileNetV2, 2-step training (feature extraction → fine-tuning) |
| 3 — XAI | `xai.py` | Accuracy/Precision/Recall/F1, Confusion Matrix, Grad-CAM heatmap |
| 4 — Reliability | `reliability.py` | Input validation, confidence thresholding (85%), SQLite/CSV logging |
| 5 — Deploy | `api.py` + `app.py` | FastAPI POST /predict, Streamlit upload UI |

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum confidence for certain predictions |
| `API_URL` | `http://localhost:8000` | FastAPI endpoint URL (for Streamlit API mode) |

---

## ⚠️ Disclaimer

This system is for **research and assistive purposes only**. It is **NOT** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
