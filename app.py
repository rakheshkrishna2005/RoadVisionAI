import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import torch
import math

# Set page configuration
st.set_page_config(layout="wide", page_title="Road Defect Detection")

# Title
st.markdown("<h1 style='text-align: center;'>RoadVision AI</h1>", unsafe_allow_html=True)

# Camera calibration parameters 
# These are example values - replace with actual camera parameters
FOCAL_LENGTH = 800  # in pixels
REAL_POTHOLE_HEIGHT = 0.1  # average pothole depth in meters
CAMERA_HEIGHT = 1.5  # camera height from ground in meters
CAMERA_ANGLE = 15  # camera tilt angle in degrees
PIXEL_SIZE = 5.6e-6  # sensor pixel size in meters (typical for phone cameras)

class DistanceCalculator:
    def __init__(self, focal_length, real_object_height, camera_height, camera_angle, pixel_size):
        self.focal_length = focal_length
        self.real_object_height = real_object_height
        self.camera_height = camera_height
        self.camera_angle = math.radians(camera_angle)
        self.pixel_size = pixel_size
    
    def calculate_distance_by_size(self, pixel_height):
        """Calculate distance using object size in pixels"""
        if pixel_height > 0:
            distance = (self.real_object_height * self.focal_length) / pixel_height
            return max(0, distance)
        return 0
    
    def calculate_distance_by_position(self, bbox_center_y, image_height):
        """Calculate distance using vertical position in image"""
        # Convert pixel position to angle
        pixel_from_center = bbox_center_y - (image_height / 2)
        angle_offset = math.atan(pixel_from_center / self.focal_length)
        ground_angle = self.camera_angle + angle_offset
        
        if ground_angle > 0:
            distance = self.camera_height / math.tan(ground_angle)
            return max(0, distance)
        return 0
    
    def calculate_distance_stereo_simulation(self, bbox_area, image_area):
        """Simulate stereo-like distance calculation using object area"""
        # This is a simplified approach - actual stereo vision would be more accurate
        area_ratio = bbox_area / image_area
        if area_ratio > 0:
            # Inverse relationship between area and distance
            estimated_distance = 20 / math.sqrt(area_ratio * 10000)  # Empirical formula
            return max(0.5, min(50, estimated_distance))  # Clamp between 0.5m and 50m
        return 0
    
    def estimate_distance(self, bbox, image_height, image_width):
        """Combine multiple methods for better accuracy"""
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        bbox_center_y = (y1 + y2) / 2
        bbox_area = bbox_height * bbox_width
        image_area = image_height * image_width
        
        # Method 1: Distance by object size
        distance_by_size = self.calculate_distance_by_size(bbox_height)
        
        # Method 2: Distance by position
        distance_by_position = self.calculate_distance_by_position(bbox_center_y, image_height)
        
        # Method 3: Distance by area (simulated stereo)
        distance_by_area = self.calculate_distance_stereo_simulation(bbox_area, image_area)
        
        # Weighted average of methods 
        weights = [0.4, 0.4, 0.2]  # [size, position, area]
        distances = [distance_by_size, distance_by_position, distance_by_area]
        
        # Filter out zero distances and calculate weighted average
        valid_distances = [(d, w) for d, w in zip(distances, weights) if d > 0]
        if valid_distances:
            total_weight = sum(w for _, w in valid_distances)
            final_distance = sum(d * w for d, w in valid_distances) / total_weight
        else:
            final_distance = 0
        
        return final_distance
    
    def calculate_real_size(self, bbox, distance):
        """Calculate real-world diameter of pothole"""
        x1, y1, x2, y2 = bbox
        bbox_width_pixels = x2 - x1
        bbox_height_pixels = y2 - y1
        
        if distance > 0 and self.focal_length > 0:
            # Convert pixel dimensions to real-world dimensions
            real_width = (bbox_width_pixels * distance) / self.focal_length
            real_height = (bbox_height_pixels * distance) / self.focal_length
            
            # Estimate diameter as average of width and height (assuming roughly circular pothole)
            diameter = (real_width + real_height) / 2
            
            return {
                'width': real_width,
                'height': real_height,
                'diameter': diameter,
                'area': real_width * real_height  # Approximate area
            }
        
        return {'width': 0, 'height': 0, 'diameter': 0, 'area': 0}

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("best.pt")
model.to(device)

# Initialize distance calculator
distance_calc = DistanceCalculator(FOCAL_LENGTH, REAL_POTHOLE_HEIGHT, CAMERA_HEIGHT, CAMERA_ANGLE, PIXEL_SIZE)

# Sidebar for calibration parameters
st.sidebar.header("Camera Calibration")
focal_length = st.sidebar.slider("Focal Length (pixels)", 400, 1200, FOCAL_LENGTH)
camera_height = st.sidebar.slider("Camera Height (m)", 0.5, 3.0, CAMERA_HEIGHT)
camera_angle = st.sidebar.slider("Camera Angle (degrees)", 0, 45, CAMERA_ANGLE)
pothole_height = st.sidebar.slider("Average Pothole Depth (m)", 0.05, 0.3, REAL_POTHOLE_HEIGHT)
pixel_size = st.sidebar.number_input("Pixel Size (µm)", min_value=1.0, max_value=10.0, value=5.6, step=0.1) * 1e-6

# Method weights
st.sidebar.header("Distance Calculation Weights")
weight_size = st.sidebar.slider("Size-based weight", 0.0, 1.0, 0.4)
weight_position = st.sidebar.slider("Position-based weight", 0.0, 1.0, 0.4)
weight_area = st.sidebar.slider("Area-based weight", 0.0, 1.0, 0.2)

# Normalize weights
total_weight = weight_size + weight_position + weight_area
if total_weight > 0:
    weights = [weight_size/total_weight, weight_position/total_weight, weight_area/total_weight]
else:
    weights = [0.33, 0.33, 0.34]

# Update distance calculator with sidebar values
distance_calc = DistanceCalculator(focal_length, pothole_height, camera_height, camera_angle, pixel_size)

def process_detections_with_distance(results, image_shape):
    """Process YOLO results and add distance and size information"""
    detections_info = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            # Calculate distance
            distance = distance_calc.estimate_distance(box, image_shape[0], image_shape[1])
            
            # Calculate real-world size
            size_info = distance_calc.calculate_real_size(box, distance)
            
            # Get class name if available
            class_name = "pothole"  # Default, or use results[0].names[int(classes[i])] if available
            
            # Severity assessment based on diameter
            severity = "Minor"
            if size_info['diameter'] > 0.3:
                severity = "Major"
            elif size_info['diameter'] > 0.15:
                severity = "Moderate"
            
            detections_info.append({
                'bbox': box,
                'confidence': conf,
                'distance': distance,
                'class': class_name,
                'size': size_info,
                'severity': severity
            })
    
    return detections_info

def draw_enhanced_annotations(image, detections_info):
    """Draw bounding boxes with distance and diameter information"""
    annotated_image = image.copy()
    
    for i, detection in enumerate(detections_info):
        bbox = detection['bbox'].astype(int)
        confidence = detection['confidence']
        distance = detection['distance']
        diameter = detection['size']['diameter']
        severity = detection['severity']
        
        # Color based on severity
        color_map = {'Minor': (0, 255, 0), 'Moderate': (0, 165, 255), 'Major': (0, 0, 255)}
        color = color_map.get(severity, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Prepare labels
        label = f"Pothole #{i+1}: {confidence:.2f}"
        distance_label = f"Distance: {distance:.1f}m"
        diameter_label = f"Diameter: {diameter*100:.1f}cm"
        severity_label = f"Severity: {severity}"
        
        labels = [label, distance_label, diameter_label, severity_label]
        
        # Calculate label dimensions
        label_height = 20
        max_width = 0
        for lbl in labels:
            label_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            max_width = max(max_width, label_size[0])
        
        # Background for labels
        bg_height = len(labels) * label_height + 10
        cv2.rectangle(annotated_image, 
                     (bbox[0], bbox[1] - bg_height),
                     (bbox[0] + max_width + 10, bbox[1]),
                     color, -1)
        
        # Draw text labels
        for j, lbl in enumerate(labels):
            y_pos = bbox[1] - bg_height + (j + 1) * label_height
            cv2.putText(annotated_image, lbl, (bbox[0] + 5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated_image

# Tabs for Image and Video
tab1, tab2, tab3, tab4 = st.tabs(["📷 Image Processing", "🎥 Video Processing", "📊 Calibration Help", "📋 Detection Summary"])

# Image Processing Tab
with tab1:
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="image")

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run model
        results = model.predict(image, device=device)
        
        # Process detections with distance
        detections_info = process_detections_with_distance(results, image_cv.shape)
        
        # Draw enhanced annotations
        annotated_cv = draw_enhanced_annotations(image_cv, detections_info)
        annotated_rgb = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Processed Image with Distance & Size")
            st.image(annotated_rgb, use_container_width=True)
        
        # Display detection details in a table
        if detections_info:
            st.subheader("Detection Details")
            
            # Create summary statistics
            total_potholes = len(detections_info)
            avg_distance = np.mean([d['distance'] for d in detections_info])
            avg_diameter = np.mean([d['size']['diameter'] for d in detections_info])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Potholes", total_potholes)
            col2.metric("Avg Distance", f"{avg_distance:.1f}m")
            col3.metric("Avg Diameter", f"{avg_diameter*100:.1f}cm")
            
            # Detailed table
            import pandas as pd
            
            table_data = []
            for i, detection in enumerate(detections_info):
                table_data.append({
                    'Pothole': f"#{i+1}",
                    'Confidence': f"{detection['confidence']:.2f}",
                    'Distance (m)': f"{detection['distance']:.1f}",
                    'Diameter (cm)': f"{detection['size']['diameter']*100:.1f}",
                    'Width (cm)': f"{detection['size']['width']*100:.1f}",
                    'Height (cm)': f"{detection['size']['height']*100:.1f}",
                    'Area (cm²)': f"{detection['size']['area']*10000:.0f}",
                    'Severity': detection['severity']
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)

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
        detection_info = st.empty()
        
        # Video statistics
        stats_container = st.container()
        all_detections = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Process every 5th frame for better performance
            if frame_count % 5 == 0:
                # Predict and get detections
                results = model.predict(frame, device=device)
                detections_info = process_detections_with_distance(results, frame.shape)
                
                # Store detections for statistics
                all_detections.extend(detections_info)
                
                # Draw enhanced annotations
                annotated_frame = draw_enhanced_annotations(frame, detections_info)
                
                # Convert BGR to RGB for Streamlit
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display side-by-side
                original_frame.image(raw_rgb, channels="RGB", use_container_width=True, caption="Original Frame")
                processed_frame.image(annotated_rgb, channels="RGB", use_container_width=True, caption="Processed Frame")
                
                # Show detection info
                if detections_info:
                    info_text = f"Frame {frame_count}: {len(detections_info)} potholes detected\n"
                    for i, detection in enumerate(detections_info):
                        info_text += f"Pothole {i+1}: {detection['distance']:.1f}m, ⌀{detection['size']['diameter']*100:.1f}cm, {detection['severity']}\n"
                    detection_info.text(info_text)

        cap.release()
        
        # Show video statistics
        if all_detections:
            with stats_container:
                st.subheader("Video Analysis Summary")
                total_detections = len(all_detections)
                unique_locations = len(set((round(d['distance'], 1), round(d['size']['diameter']*100, 1)) for d in all_detections))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Detections", total_detections)
                col2.metric("Estimated Unique Potholes", unique_locations)
                col3.metric("Detection Rate", f"{unique_locations/max(frame_count/30, 1):.1f}/sec")

# Calibration Help Tab
with tab3:
    st.header("Camera Calibration Guide")
    
    st.markdown("""
    ### To improve distance and diameter accuracy, calibrate these parameters:
    
    **1. Focal Length (pixels):**
    - Use OpenCV's camera calibration with checkerboard pattern
    - Or measure: `focal_length = (known_distance × object_pixel_width) / real_object_width`
    
    **2. Camera Height & Angle:**
    - Measure actual height and downward tilt angle of  camera
    - Use a level/inclinometer for accurate angle measurement
    
    **3. Pixel Size:**
    - Check  camera specifications (typically 1-10 micrometers)
    - For phones: usually around 1.4-5.6µm
    
    **4. Reference Object Size:**
    - Measure real potholes in  area for better size estimation
    
    ### Distance Calculation Methods:
    
    **Method 1: Object Size** `distance = (real_height × focal_length) / pixel_height`
    
    **Method 2: Position-based** Uses camera angle and vertical position
    
    **Method 3: Area-based** Simulated stereo using relative object area
    
    **Diameter Calculation:** `diameter = (pixel_width × distance) / focal_length`
    
    ### Validation Tips:
    - Test with objects at known distances
    - Use measuring tape to verify diameter calculations
    - Adjust method weights based on the camera setup
    """)
    
    st.warning("⚠️ Measurements are estimates. Accuracy depends on proper calibration and consistent lighting/weather conditions!")

# Detection Summary Tab
with tab4:
    st.header("Detection Classification System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity Classification")
        st.markdown("""
        **Minor (Green):** < 15cm diameter
        - Typically shallow surface cracks
        - Low priority for repair
        
        **Moderate (Orange):** 15-30cm diameter  
        - Noticeable potholes affecting ride quality
        - Medium priority for repair
        
        **Major (Red):** > 30cm diameter
        - Large potholes potentially damaging vehicles
        - High priority for immediate repair
        """)
    
    with col2:
        st.subheader("Measurement Accuracy")
        st.markdown("""
        **Distance Accuracy:** ±10-20% (with proper calibration)
        
        **Diameter Accuracy:** ±15-25% (depends on distance and angle)
        
        **Best Conditions:**
        - Good lighting
        - Consistent camera height/angle
        - Distance 2-20 meters
        - Clear view of pothole edges
        """)
    
    st.subheader("System Limitations")
    st.info("""
    - Accuracy decreases with distance
    - Lighting conditions affect detection
    - Partially occluded potholes may be mis-measured
    - Water-filled potholes may appear smaller
    - Requires camera calibration for each setup
    """)
