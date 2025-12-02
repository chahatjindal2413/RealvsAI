import streamlit as st
import torch
from utils import load_model, preprocess

st.set_page_config(page_title="AI vs Real Image Classifier", layout="centered")

st.title("🌀 AI-Generated vs Real Image Classifier")
st.subheader("Upload an image and get prediction with your trained ML model")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

model = load_model("model.pth")

class_names = ["Real Image", "AI-Generated Fake"]

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    img_tensor = preprocess(uploaded_file)

    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.softmax(outputs, dim=1)
        conf, predicted = torch.max(prob, 1)

    st.write("### 🔍 Prediction:")
    st.success(f"**{class_names[predicted]}**")
    
    st.write("### Confidence Score:")
    st.info(f"{conf.item()*100:.2f}%")
