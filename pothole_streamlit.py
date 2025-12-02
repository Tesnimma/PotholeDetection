"""
Pothole Segmentation Web UI with Streamlit
A clean Streamlit interface for pothole detection and segmentation
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Pothole Detection & Segmentation",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained YOLO model"""
    model_paths = [
        'runs/segment/train/weights/best.pt',
        'best.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error(
            f"Model file not found. Please ensure 'best.pt' exists in one of these locations:\n"
            f"  - {model_paths[0]}\n"
            f"  - {model_paths[1]}"
        )
        st.stop()
    
    try:
        model = YOLO(model_path)
        return model, model_path
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load model
model, model_path = load_model()

# Header
st.markdown("""
    <div class="main-header">
        <h1>🕳️ Pothole Detection & Segmentation</h1>
        <p>Upload an image to detect and segment potholes using YOLOv8</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Model loaded from: `{model_path}`")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Adjust the confidence threshold for detections. Lower values = more detections (may include false positives). Higher values = fewer detections (only high-confidence detections)."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.text("Model: YOLOv8 Segmentation")
    st.text("Task: Pothole Detection")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing potholes to analyze"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    
    if uploaded_file is not None:
        # Process button
        if st.button("🔍 Detect Potholes", type="primary", use_container_width=True):
            try:
                with st.spinner("Processing image..."):
                    # Convert PIL Image to numpy array
                    image_array = np.array(image)
                    
                    # Ensure image is in the right format
                    if image_array.dtype != np.uint8:
                        if image_array.max() <= 1.0:
                            image_array = (image_array * 255).astype(np.uint8)
                        else:
                            image_array = image_array.astype(np.uint8)
                    
                    # Ensure 3 channels (RGB)
                    if len(image_array.shape) == 2:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                    
                    # Run inference
                    results = model.predict(
                        image_array,
                        conf=confidence_threshold,
                        save=False,
                        verbose=False,
                        device='cpu'
                    )
                    result = results[0]
                    
                    # Get original image (already in RGB)
                    original_img = image_array.copy()
                    
                    # Get annotated image
                    annotated_img = result.plot()
                    
                    # Get detection count
                    num_detections = len(result.boxes) if result.boxes is not None else 0
                    
                    # Display results
                    st.success(f"✅ Processing complete! Found {num_detections} pothole(s)")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Original", "Segmented", "Side by Side"])
                    
                    with tab1:
                        st.image(original_img, caption="Original Image", use_container_width=True)
                    
                    with tab2:
                        st.image(annotated_img, caption="Segmented Image with Detections", use_container_width=True)
                    
                    with tab3:
                        # Create side-by-side comparison
                        h = max(original_img.shape[0], annotated_img.shape[0])
                        w1 = int(original_img.shape[1] * h / original_img.shape[0]) if original_img.shape[0] > 0 else original_img.shape[1]
                        w2 = int(annotated_img.shape[1] * h / annotated_img.shape[0]) if annotated_img.shape[0] > 0 else annotated_img.shape[1]
                        
                        orig_resized = cv2.resize(original_img, (w1, h))
                        annot_resized = cv2.resize(annotated_img, (w2, h))
                        comparison = np.hstack([orig_resized, annot_resized])
                        
                        st.image(comparison, caption="Original vs Segmented", use_container_width=True)
                    
                    # Display detection summary
                    st.markdown("---")
                    st.subheader("📈 Detection Summary")
                    
                    if num_detections > 0:
                        st.success(f"**Number of potholes detected:** {num_detections}")
                        
                        # Create a table with detection details
                        detection_data = []
                        for i, box in enumerate(result.boxes, 1):
                            conf = box.conf.item()
                            detection_data.append({
                                "Pothole #": i,
                                "Confidence": f"{conf:.1%}"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("No potholes detected in this image. Try lowering the confidence threshold.")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
    else:
        st.info("👆 Please upload an image to get started")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>This model uses YOLOv8 for pothole detection and segmentation.</p>
        <p>Adjust the confidence threshold in the sidebar to control detection sensitivity.</p>
    </div>
""", unsafe_allow_html=True)

