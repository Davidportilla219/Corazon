import streamlit as st
import joblib
import numpy as np

# Cargar modelo y scaler
model = joblib.load("svc_model.jb")
scaler = joblib.load("scaler.jb")

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
        st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.es%2Ffotos-vectores-gratis%2Fpersona-feliz&psig=AOvVaw0ILTB-AXf3pLY6QHiiacZr&ust=1759438731007000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCJiPitbzg5ADFQAAAAAdAAAAABAE", 
                 caption="Persona feliz")
    else:
        st.error("Sufrirá del corazón 💔")
        st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.ecured.cu%2FMuerte&psig=AOvVaw3TaRjR41snFYdSUPTFnmX7&ust=1759438762399000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCPDKgubzg5ADFQAAAAAdAAAAABAL", 
                 caption="Problema cardíaco")
