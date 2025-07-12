import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import torch

# Set page configuration
st.set_page_config(layout="wide", page_title="Road Defect Detection")

# Title
st.markdown("<h1 style='text-align: center;'>RoadVision AI​</h1>", unsafe_allow_html=True)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("best.pt")
model.to(device)

# Tabs for Image and Video
tab1, tab2 = st.tabs(["📷 Image Processing", "🎥 Video Processing"])

# Image Processing Tab
with tab1:
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image")

    if uploaded_image:
        image = Image.open(uploaded_image)

        # Run model
        results = model.predict(image, device=device)
        annotated = results[0].plot()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Processed Image")
            st.image(annotated, channels="BGR", use_container_width=True)

# Video Processing Tab
with tab2:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        col1, col2 = st.columns(2)
        original_frame = col1.empty()
        processed_frame = col2.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict and annotate
            results = model.predict(frame, device=device)
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display side-by-side
            original_frame.image(raw_rgb, channels="RGB", use_container_width=True, caption="Original Frame")
            processed_frame.image(annotated_rgb, channels="RGB", use_container_width=True, caption="Processed Frame")

        cap.release()
