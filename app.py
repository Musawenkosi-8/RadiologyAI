import streamlit as st
from PIL import Image
import numpy as np
import cv2

from utils import load_model, preprocess_image, load_labels, apply_gradcam

st.set_page_config(page_title="RadiologyAI", layout="centered")
st.title("🧠 RadiologyAI – Chest X-ray Diagnosis")

model = load_model("model.pt")
labels_dict = load_labels("labels.json")

language = st.selectbox("Choose Language", ["en", "fr", "zu"])

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    image_tensor = preprocess_image(image)
    grayscale_cam, class_idx = apply_gradcam(model, image_tensor)

    label = labels_dict[str(class_idx)][language]
    st.markdown(f"### 🩺 Prediction: **{label}**")

    cam_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    cam_overlay = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    cam_overlay = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(0.5 * cam_img + 0.5 * cam_overlay)

    st.markdown("### 🔬 Grad-CAM Explanation")
    st.image(heatmap, use_column_width=True)
