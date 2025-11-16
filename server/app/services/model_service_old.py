import os
import sys
import joblib
import numpy as np
from app.config.config import Config

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PROJECT_ROOT)

from ml.scripts.random_forest.features import extract_features

THRESHOLD_ROTTEN = 0.5

# Pastikan model ada
if not os.path.exists(Config.MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file tidak ditemukan di: {Config.MODEL_PATH}")

# Load model hanya sekali
model = joblib.load(Config.MODEL_PATH)


def predict_image(image_path, use_threshold=False):
    """
    Prediksi gambar mangga menggunakan Random Forest.
    """
    try:
        features = extract_features(image_path)
        features = np.expand_dims(features, axis=0)

        pred_proba = model.predict_proba(features)[0]
        classes = model.classes_.tolist()

        healthy_idx = classes.index("mango_healthy")
        rotten_idx = classes.index("mango_rotten")

        # Gunakan threshold manual (opsional)
        if use_threshold:
            if pred_proba[rotten_idx] >= THRESHOLD_ROTTEN:
                pred_label = "mango_rotten"
            else:
                pred_label = "mango_healthy"
            method = f"threshold_{THRESHOLD_ROTTEN}"
        else:
            pred_label = model.predict(features)[0]
            method = "default_0.5"

        confidence = float(max(pred_proba))

        return {
            "label": pred_label,
            "probabilities": {
                "mango_healthy": float(pred_proba[healthy_idx]),
                "mango_rotten": float(pred_proba[rotten_idx]),
            },
            "confidence": confidence,
            "method": method,
            "threshold_info": {
                "rotten_threshold": THRESHOLD_ROTTEN,
                "healthy_implied": 1 - THRESHOLD_ROTTEN,
            },
        }

    except Exception as e:
        raise RuntimeError(f"Gagal melakukan prediksi: {e}")
