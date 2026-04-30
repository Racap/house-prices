import streamlit as st
from predict import predict_price
from preprocess import MODEL_FEATURES

SQFT_PER_M2 = 10.7639

st.title("🏠 Predictor de precios de vivienda")
st.write("Aplicación desarrollada para AI Foundations — Fundació URV")
st.caption("Modelo: RandomForestRegressor entrenado con el dataset House Prices (Kaggle).")

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
        prediction = predict_price(feature_values)
        st.success(f"Precio estimado: ${prediction:,.2f}")
        st.info("La predicción es orientativa y depende de las variables incluidas en el modelo.")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except KeyError as exc:
        st.error(f"Falta una variable requerida por el modelo: {exc}")
    except Exception as exc:
        st.error(f"Error al generar la predicción: {exc}")

st.divider()
st.write("Variables usadas por el modelo:")
st.code(", ".join(MODEL_FEATURES))