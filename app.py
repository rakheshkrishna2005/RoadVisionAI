import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image

st.set_page_config(layout="centered", page_title="Road Defect Detection")

st.markdown(
    "<h1 style='text-align: center;'>RoadVision AI​</h1>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "mp4", "mov", "avi"])

model = YOLO("best.pt")

if uploaded_file:
    file_type = uploaded_file.type

    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        results = model.predict(image)
        annotated = results[0].plot()
        st.image(annotated, channels="BGR", use_container_width=True)

    elif file_type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        cap.release()
