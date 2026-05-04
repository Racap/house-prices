from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "baseline": ROOT_DIR / "models" / "house_price_model.joblib",
    "tuned": ROOT_DIR / "models" / "house_price_tuned.joblib",
}


def load_artifact(model_name: str = "baseline") -> dict:
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modelo no reconocido: {model_name}")

    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo '{model_name}'. "
            "Ejecuta primero: python3 src/train.py"
        )
    return joblib.load(model_path)


def predict_price(input_features: Dict[str, float], model_name: str = "baseline") -> float:
    artifact = load_artifact(model_name)
    model = artifact["model"]
    features = artifact["features"]

    sample = pd.DataFrame([{feature: input_features[feature] for feature in features}])
    prediction = model.predict(sample)[0]
    return float(prediction)
