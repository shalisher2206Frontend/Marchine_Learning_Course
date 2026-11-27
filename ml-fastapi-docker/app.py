import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.title("🌸 Iris Classifier")
st.write("Введи размеры цветка и нажми Predict")

# Два столбца, чтобы поля стояли красиво
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.5)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        res = requests.post(API_URL, json=data)
        result = res.json()
        st.success(f"Species: **{result['species']}**")
    except requests.exceptions.ConnectionError:
        st.error("API не отвечает. Запущен ли Docker-контейнер?")