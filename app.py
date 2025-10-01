import streamlit as st
import joblib
import numpy as np

# Cargar modelo y scaler
model = joblib.load("Spacer.jb")
scaler = joblib.load("Scaler.jb")

# Título y subtítulo
st.title("Detector de problemas cardiacos")
st.subheader("Elaborado por Unab 2025")

# Sliders para la entrada de datos
edad = st.slider("Edad", min_value=25, max_value=80, value=55, step=1)
colesterol = st.slider("Colesterol", min_value=120, max_value=600, value=242, step=2)

# Preparar datos para el modelo
features = np.array([[edad, colesterol]])
features_scaled = scaler.transform(features)

# Botón de predicción
if st.button("Predecir"):
    prediction = model.predict(features_scaled)[0]

    if prediction == 0:
        st.success("No sufrirá del corazón ❤️")
        st.image("https://img.freepik.com/vector-gratis/hombre-feliz-dibujos-animados_52683-56651.jpg", 
                 caption="Persona feliz")
    else:
        st.error("Sufrirá del corazón 💔")
        st.image("https://www.topdoctors.mx/files/Image/large/f54aa2d6eae7245e58f87d4f62a4a16b.jpg", 
                 caption="Problema cardíaco")
