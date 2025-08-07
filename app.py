import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import torch
import time
import os

CLASSES = ['Pothole']

# Configure Streamlit page settings
st.set_page_config(initial_sidebar_state='expanded')

# Load and apply custom CSS styling
with open('main.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def detect_objects(image, model, classes, conf):
    """
    Perform object detection on an input image using the YOLO model.
    
    Args:
        image: Input image array
        model: Loaded YOLO model instance
        classes: List of class indices to detect
        conf: Confidence threshold for detections
    
    Returns:
        tuple: (annotated_image, detected_class_indices)
            - annotated_image: Image with detection boxes drawn
            - detected_class_indices: List of class indices for detected objects
    """
    results = model(image, conf=conf, classes=classes)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
    class_indices = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
    annotated_image = image.copy()
    
    # Conversion factor: 1 pixel = 0.25 cm
    PIXEL_TO_CM = 0.75
    
    for box, cls_idx in zip(boxes, class_indices):
        x1, y1, x2, y2 = map(int, box)
        color = (255, 0, 0)  # Red box (RGB format: Red=255, Green=0, Blue=0)
        
        # Calculate bounding box dimensions in pixels
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Convert to centimeters
        width_cm = width_px * PIXEL_TO_CM
        height_cm = height_px * PIXEL_TO_CM
        
        # Calculate diameter (average of width and height)
        diameter_cm = (width_cm + height_cm) / 2
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)  # thickness=2
        
        # Create label with class name and diameter
        label = f"{CLASSES[int(cls_idx)]} {diameter_cm:.0f}cm"
        
        # Draw label background and text
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    
    return annotated_image, class_indices.tolist()

def is_cuda_available():
    """
    Check if CUDA (GPU acceleration) is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()

def main():
    """
    Main application function that sets up the Streamlit interface and handles
    the object detection pipeline for both webcam and uploaded video inputs.
    """
    # Initialize session state variables for metrics tracking
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = 0
    if 'tracked_objects' not in st.session_state:
        st.session_state.tracked_objects = 0
    if 'detected_classes' not in st.session_state:
        st.session_state.detected_classes = 0

    # Set up main page title and divider
    st.markdown("<h1 style='text-align: center;'>RoadVision AI", unsafe_allow_html=True)
    st.markdown("---")
    
    # Configure sidebar settings and styling
    st.sidebar.title("⚙️ Settings")
    st.markdown("""
                <style>
                .stButton > button {
                    width: 100%;
                }
                </style>""", 
                unsafe_allow_html=True)
    
    # Check and display CUDA availability
    cuda_available = is_cuda_available()
    
    # Create settings container in sidebar
    st.sidebar.markdown('<div class="settings-container">', unsafe_allow_html=True)
    
    # Add user configuration options
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.sidebar.markdown("---")
    enable_gpu = st.sidebar.checkbox("⚡ Enable GPU", value=False, disabled=not cuda_available)    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add input source selection options
    use_webcam = st.sidebar.button("Use Webcam", use_container_width=True)
    selected_classes = st.sidebar.multiselect("Select Classes", CLASSES, default=['Pothole'])
    
    # Display CUDA availability warning if needed
    if not cuda_available:
        st.sidebar.warning("CUDA is not available. GPU acceleration is disabled.")
        st.sidebar.info("To enable GPU acceleration, make sure you have a CUDA-capable GPU and PyTorch is installed with CUDA support.")
    
    # Add video upload option
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    # Convert selected class names to their corresponding indices
    class_indices = [CLASSES.index(cls) for cls in selected_classes]

    # Initialize YOLO model with appropriate device (GPU/CPU)
    model = YOLO('pothole.pt')
    if enable_gpu and cuda_available:
        model.to('cuda')
        st.sidebar.success("GPU enabled successfully!")
    else:
        model.to('cpu')
        st.sidebar.info("Using CPU for processing.")
    
    # Create video display container
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Create metrics display layout
    kpi_col1, kpi_col2 = st.columns(2)
    tracked_objects_metric = kpi_col1.empty()
    frame_rate_metric = kpi_col2.empty()

    # Initialize metrics with default values
    tracked_objects_metric.metric("Tracked Objects", "0")
    frame_rate_metric.metric("Frame Rate", "0.00 FPS")

    # Create placeholder for object count table
    object_count_placeholder = st.empty()

    # Add CSS styling for the detection results table
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

    # Store processed frames for potential future use
    processed_frames = []

    # Handle webcam input
    if use_webcam:
        cap = cv2.VideoCapture(0)
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update metrics every second
                frame_rate = frame_count / elapsed_time
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

            if not use_webcam:
                break

        cap.release()

    # Handle uploaded video input
    elif uploaded_video is not None:
        # Create temporary file for video processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        vf = cv2.VideoCapture(tfile.name)
        frame_count = 0
        start_time = time.time()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame, detected_classes = detect_objects(frame, model, class_indices, conf_threshold)
            video_placeholder.image(annotated_frame, channels="RGB")
            processed_frames.append(annotated_frame)

            # Update performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.5:  # Update metrics every half second
                frame_rate = frame_count / elapsed_time
                tracked_objects_metric.metric("Tracked Objects", str(len(detected_classes)))
                frame_rate_metric.metric("Frame Rate", f"{frame_rate:.2f} FPS")
                frame_count = 0
                start_time = time.time()

            # Update and display object count table
            unique_classes, counts = np.unique(detected_classes, return_counts=True)
            object_data = [{"Class": CLASSES[int(cls)], "Count": count} for cls, count in zip(unique_classes, counts)]
            object_count_placeholder.markdown(
                "<table class='detected-object-table'>" +
                "<tr><th>Class</th><th>Count</th></tr>" +
                "".join([f"<tr><td>{item['Class']}</td><td>{item['Count']}</td></tr>" for item in object_data]) +
                "</table>",
                unsafe_allow_html=True
            )

        vf.release()

    # Update final metrics display
    st.markdown(f"""
    <script>
        document.getElementById('tracked-objects').innerText = "{st.session_state.tracked_objects}";
        document.getElementById('frame-rate').innerText = "{st.session_state.frame_rate:.2f} FPS";
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()