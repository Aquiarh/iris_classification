import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Iris Species Classifier", layout="centered")

st.title("🌸 Iris Flower Classification Portal")
st.write("Enter flower dimensions to predict species classification.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('iris_model.pkl')

model = load_model()

col1, col2 = st.columns(2)
with col1:
    sepal_len = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    sepal_wid = st.slider("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
with col2:
    petal_len = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1)
    petal_wid = st.slider("Petal Width (cm)", min_value=0.1, max_value=3.0, value=1.3, step=0.1)

if st.button("Predict Species"):
    features = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    st.success(f"Predicted Species: **{prediction.upper()}**")
    
    st.subheader("Prediction Confidence")
    for cls, prob in zip(model.classes_, probabilities):
        st.write(f"- **{cls.capitalize()}**: {prob * 100:.2f}%")