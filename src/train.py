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
TUNED_MODEL_PATH = ROOT_DIR / "models" / "house_price_tuned.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics" / "model_comparison.csv"

BASELINE_PARAMS = {
    "n_estimators": 300,
}

TUNED_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "max_features": "log2",
}


def build_model(params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def save_artifact(model: Pipeline, path: Path, version: str, params: dict) -> None:
    joblib.dump(
        {
            "model": model,
            "features": MODEL_FEATURES,
            "target": TARGET_COLUMN,
            "version": version,
            "params": params,
        },
        path,
    )


def train_and_evaluate() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_model = build_model(BASELINE_PARAMS)
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    save_artifact(
        baseline_model,
        MODEL_PATH,
        "baseline_random_forest",
        BASELINE_PARAMS,
    )

    tuned_model = build_model(TUNED_PARAMS)
    tuned_model.fit(X_train, y_train)
    tuned_metrics = evaluate_model(tuned_model, X_test, y_test)
    save_artifact(
        tuned_model,
        TUNED_MODEL_PATH,
        "tuned_random_forest",
        TUNED_PARAMS,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "version": "baseline_random_forest",
                "rmse": round(baseline_metrics["rmse"], 4),
                "mae": round(baseline_metrics["mae"], 4),
                "r2": round(baseline_metrics["r2"], 4),
            },
            {
                "version": "tuned_random_forest",
                "rmse": round(tuned_metrics["rmse"], 4),
                "mae": round(tuned_metrics["mae"], 4),
                "r2": round(tuned_metrics["r2"], 4),
            }
        ]
    )
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("Entrenamiento completado.")
    print(f"Modelo baseline guardado en: {MODEL_PATH}")
    print(f"Modelo tuned guardado en: {TUNED_MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print(
        "Baseline "
        f"RMSE: {baseline_metrics['rmse']:.2f} | "
        f"MAE: {baseline_metrics['mae']:.2f} | "
        f"R2: {baseline_metrics['r2']:.4f}"
    )
    print(
        "Tuned "
        f"RMSE: {tuned_metrics['rmse']:.2f} | "
        f"MAE: {tuned_metrics['mae']:.2f} | "
        f"R2: {tuned_metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    train_and_evaluate()
