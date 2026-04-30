from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "house_price_model.joblib"


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Ejecuta primero: python3 src/train.py"
        )
    return joblib.load(MODEL_PATH)


def predict_price(input_features: Dict[str, float]) -> float:
    artifact = load_artifact()
    model = artifact["model"]
    features = artifact["features"]

    sample = pd.DataFrame([{feature: input_features[feature] for feature in features}])
    prediction = model.predict(sample)[0]
    return float(prediction)
