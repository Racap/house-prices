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

## Jupyter Lab

Los cuadernos de análisis y modelado están en `notebooks/`. Para abrirlos con **Jupyter Lab**:

1. Instala dependencias (incluye `jupyterlab`; ver `requirements.txt`).
2. Desde la **raíz del repositorio** (así las rutas a `data/` y otros recursos coinciden con el código de los notebooks):

```bash
jupyter lab
```

3. En el navegador, abre los archivos `.ipynb` dentro de `notebooks/`.

**Notas:**

- Si usas un entorno virtual, actívalo antes de `pip install` y de `jupyter lab`.
- Jupyter Lab suele abrirse solo en `http://localhost:8888`; si el puerto está ocupado, la terminal mostrará otra URL con token.
- Alternativa clásica: `jupyter notebook` (interfaz distinta, mismos notebooks).

## Entrenamiento

Desde la raíz del repositorio:

```bash
python3 src/train.py
```

Este comando:

- entrena un modelo de regresión (`RandomForestRegressor`)
- guarda el modelo baseline en `models/house_price_model.joblib`
- guarda el modelo tuneado en `models/house_price_tuned.joblib`
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
