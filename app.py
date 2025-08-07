import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

CLASSES = ['pothole']

st.set_page_config(initial_sidebar_state='expanded')

with open('main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def is_cuda_available():
    return torch.cuda.is_available()

def main():
    if 'unique_track_ids' not in st.session_state:
        st.session_state.unique_track_ids = set()

    st.markdown("<h1 style='text-align: center;'>Pothole Detection and Tracking", unsafe_allow_html=True)
    st.markdown("---")
    
    st.sidebar.title("⚙️ Settings")
    st.markdown("""
                <style>
                .stButton > button {
                    width: 100%;
                }
                </style>""", 
                unsafe_allow_html=True)
    
    cuda_available = is_cuda_available()
    
    st.sidebar.markdown('<div class="settings-container">', unsafe_allow_html=True)
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.sidebar.markdown("---")
    enable_gpu = st.sidebar.checkbox("🤖 Enable GPU", value=False, disabled=not cuda_available)    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    selected_classes = st.sidebar.multiselect("Select Classes", CLASSES, default=['pothole'])
    
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")
        st.sidebar.info("To enable GPU acceleration, make sure you have a CUDA-capable GPU and PyTorch is installed with CUDA support.")
    
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    class_indices = [CLASSES.index(cls) for cls in selected_classes]

    model = YOLO('pothole.pt')
    if enable_gpu and cuda_available:
        model.to('cuda')
        st.sidebar.success("GPU enabled successfully!")
        device = 0
    else:
        model.to('cpu')
        st.sidebar.info("Using CPU for processing.")
        device = 'cpu'
    
    deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_age=70)
    
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    object_count_placeholder = st.empty()

    st.markdown("""
    <style>
    .detected-object-table {
        width: 100%;
        border-collapse: collapse;
        text-align: center;
    }
    .detected-object-table th, .detected-object-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .detected-object-table th {
        background-color: var(--background-color);
    }
    .detected-object-table tr:nth-child(even) {
        background-color: var(--background-color);
    }
    </style>
    """, unsafe_allow_html=True)

    processed_frames = []

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(og_frame, device=device, classes=class_indices, conf=conf_threshold)
            
            cls = []
            xyxy = []
            conf = []
            xywh = []
            
            for result in results:
                boxes = result.boxes
                cls = boxes.cls.tolist()
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh
            
            if len(cls) > 0:
                pred_cls = np.array(cls)
                bbox_conf = conf.detach().cpu().numpy()
                bboxes_xywh = xywh.cpu().numpy()
                
                tracks = tracker.update(bboxes_xywh, bbox_conf, og_frame)
                
                for track in tracker.tracker.tracks:
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.to_tlbr()
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Use red color for bounding boxes (BGR format)
                    box_color = (0, 0, 255)  # Red color in BGR
                    
                    cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), box_color, 2)
                    
                    class_name = CLASSES[int(cls[0])] if cls else "Unknown"
                    text_color = (255, 255, 255)  # White color in BGR
                    cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)  # Changed thickness from 2 to 1
                    
                    st.session_state.unique_track_ids.add(track_id)
            
            video_placeholder.image(og_frame, channels="RGB")
            processed_frames.append(og_frame)
            
            # Continue with pothole detection

            unique_classes = np.unique(cls).astype(int) if len(cls) > 0 else []
            class_counts = {}
            for cls_id in unique_classes:
                class_name = CLASSES[cls_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Update object count table with unique pothole count
            unique_pothole_count = len(st.session_state.unique_track_ids)
            object_data = []
            for cls_name, count in class_counts.items():
                if cls_name == "pothole":
                    object_data.append({"Class": cls_name, "Count": unique_pothole_count})
                else:
                    object_data.append({"Class": cls_name, "Count": count})
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Unique Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        cap.release()

    # End of main function

if __name__ == "__main__":
    main()