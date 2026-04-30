# House Prices - AI Foundations URV

Proyecto final de la microcredencial **Artificial Intelligence Foundations** (Fundació URV).
Reto: **ML clásico - Predicción de precios de vivienda**.

## Equipo

- Adria
- Raul

## Objetivo

Construir un pipeline de aprendizaje supervisado (regresión) para predecir `SalePrice`
a partir de variables de la vivienda.

## Dataset

- Fuente: Kaggle House Prices
- Archivo: `data/train.csv`
- Registros: 1460
- Variable objetivo: `SalePrice`

## Estructura principal

- `src/`: app Streamlit y scripts de entrenamiento/predicción
- `notebooks/`: EDA, preprocesado, modelado y tuning
- `models/`: artefacto del modelo entrenado
- `artifacts/metrics/`: tabla de métricas
- `artifacts/figures/`: figuras para documentación y slides

## Requisitos

```bash
pip install -r requirements.txt
```

## Entrenamiento

Desde la raíz del repositorio:

```bash
python3 src/train.py
```

Este comando:

- entrena un modelo de regresión (`RandomForestRegressor`)
- guarda el artefacto en `models/house_price_model.joblib`
- genera métricas en `artifacts/metrics/model_comparison.csv`

## App Streamlit

```bash
streamlit run src/app.py
```

La app carga el modelo entrenado y permite introducir variables de vivienda para obtener
una predicción de precio estimado.

## Métricas

Las métricas de evaluación se guardan en `artifacts/metrics/model_comparison.csv`:

- RMSE
- MAE
- R²

## Entregables URV

- `doc.pdf`
- `slides.pdf`
- `streamlit.txt`
- `video.txt`
