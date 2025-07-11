import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

st.set_page_config(layout="centered", page_title="Road Defect Detection")

st.markdown(
    "<h1 style='text-align: center;'>AI-Driven Road Quality Monitoring and Defect Detection System​</h1>",
    unsafe_allow_html=True
)

model = YOLO("best.pt")

st.sidebar.title("Upload")

option = st.sidebar.radio("Choose input type", ("Image", "Video"))

if option == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        results = model.predict(image)
        annotated = results[0].plot()
        st.image(annotated, caption="Detected Image", use_column_width=True)

elif option == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)
        cap.release()
