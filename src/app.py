import streamlit as st
from predict import predict_price
from preprocess import MODEL_FEATURES

SQFT_PER_M2 = 10.7639

st.title("🏠 Predictor de precios de vivienda")
st.write("Aplicación desarrollada para AI Foundations — Fundació URV")
st.caption(
    "Comparativa entre RandomForestRegressor baseline y modelo con tuning de "
    "hiperparámetros."
)

st.subheader("Introduce las características de la vivienda")

feature_values = {
    "OverallQual": st.slider("Calidad general (OverallQual)", min_value=1, max_value=10, value=5),
    "GrLivArea": st.number_input(
        "Superficie habitable (m²)",
        min_value=20.0,
        max_value=600.0,
        value=140.0,
        step=1.0,
    )
    * SQFT_PER_M2,
    "GarageCars": st.slider("Capacidad del garaje (GarageCars)", min_value=0, max_value=4, value=2),
    "TotalBsmtSF": st.number_input(
        "Superficie sótano (m²)",
        min_value=0.0,
        max_value=500.0,
        value=85.0,
        step=1.0,
    )
    * SQFT_PER_M2,
    "FullBath": st.slider("Baños completos (FullBath)", min_value=0, max_value=4, value=2),
    "YearBuilt": st.number_input("Año de construcción (YearBuilt)", min_value=1872, max_value=2026, value=2000, step=1),
}

if st.button("Predecir precio"):
    try:
        baseline_prediction = predict_price(feature_values, model_name="baseline")
        tuned_prediction = predict_price(feature_values, model_name="tuned")
        difference = tuned_prediction - baseline_prediction

        st.subheader("Comparativa de predicción")
        baseline_col, tuned_col = st.columns(2)

        with baseline_col:
            st.metric(
                "Baseline Random Forest",
                f"${baseline_prediction:,.2f}",
                help="Modelo base sin ajuste de hiperparámetros.",
            )

        with tuned_col:
            st.metric(
                "Tuned Random Forest",
                f"${tuned_prediction:,.2f}",
                delta=f"${difference:,.2f} vs baseline",
                help="Modelo entrenado con los mejores hiperparámetros del tuning.",
            )

        st.info(
            "La diferencia muestra cuánto cambia la predicción del modelo tuneado "
            "respecto al baseline para las mismas variables de entrada."
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
    except KeyError as exc:
        st.error(f"Falta una variable requerida por el modelo: {exc}")
    except Exception as exc:
        st.error(f"Error al generar la predicción: {exc}")

st.divider()
st.write("Variables usadas por el modelo:")
st.code(", ".join(MODEL_FEATURES))