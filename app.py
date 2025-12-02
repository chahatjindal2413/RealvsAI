import streamlit as st
from utils import preprocess, predict

st.title("AI vs Real Image Classifier (DINOv2 + Custom Head)")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded, width=300)

    img = preprocess(uploaded)
    label, conf = predict(img)

    classes = ["Real Image", "AI-Generated Fake"]

    st.success(f"Prediction: {classes[label]}")
    st.info(f"Confidence: {conf*100:.2f}%")
