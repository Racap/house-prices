from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import MODEL_FEATURES, TARGET_COLUMN


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "train.csv"
MODEL_PATH = ROOT_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics" / "model_comparison.csv"


def train_and_evaluate() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": MODEL_FEATURES,
            "target": TARGET_COLUMN,
        },
        MODEL_PATH,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "version": "baseline_random_forest",
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r2": round(r2, 4),
            }
        ]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("Entrenamiento completado.")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
