# Detector de Problemas Cardiacos ❤️

Aplicación desarrollada con **Streamlit** que permite predecir si una persona sufrirá o no del corazón, utilizando un modelo **SVC (Support Vector Classifier)** previamente entrenado.

## 📌 Descripción
El modelo fue entrenado con las variables:
- **Edad** (entre 25 y 80 años).
- **Colesterol** (entre 120 y 600 mg/dL).

La etiqueta objetivo fue:
- **Problemas cardiacos** (`0 = No sufrirá del corazón`, `1 = Sí sufrirá del corazón`).

Los datos fueron **normalizados con MinMaxScaler** y tanto el modelo como el scaler se guardaron en archivos `.jb` con `joblib`.

## 🚀 Funcionalidades
- Interfaz gráfica con **Streamlit**.
- Entrada de datos con sliders:
  - Edad (25 a 80, valor por defecto 55).
  - Colesterol (120 a 600, valor por defecto 242).
- Predicción en tiempo real con el modelo SVC.
- Visualización de imágenes dependiendo del resultado:
  - `0` → Persona feliz 😊
  - `1` → Problema cardíaco 💔

## 📂 Archivos del proyecto
- `app.py` → Código principal de la aplicación.
- `Spacer.jb` → Modelo entrenado (SVC).
- `Scaler.jb` → Normalizador MinMaxScaler.
- `requirements.txt` → Dependencias necesarias.
- `README.md` → Documentación del proyecto.

## ▶️ Cómo ejecutar la aplicación

1. Clonar el repositorio o descargar los archivos.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
