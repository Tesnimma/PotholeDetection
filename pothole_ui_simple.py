"""
Pothole Segmentation Web UI - Simplified Version
A simplified Gradio interface that works around version compatibility issues
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio is not installed. Please install it with: pip3 install gradio")
    sys.exit(1)

# Load the trained model
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
        raise FileNotFoundError(
            f"Model file not found. Please ensure 'best.pt' exists in one of these locations:\n"
            f"  - {model_paths[0]}\n"
            f"  - {model_paths[1]}"
        )
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully! ✓")
    return model

# Load model once at startup
model = load_model()

def segment_pothole(input_image, confidence_threshold=0.25):
    """
    Perform pothole segmentation on an uploaded image
    """
    if input_image is None:
        return None, None, "Please upload an image first."
    
    try:
        # Gradio passes numpy arrays in RGB format when type="numpy"
        # YOLO can handle numpy arrays directly, but we need to ensure correct format
        if isinstance(input_image, Image.Image):
            # Convert PIL Image to numpy array (RGB)
            image_array = np.array(input_image)
        elif isinstance(input_image, np.ndarray):
            # Already a numpy array from Gradio (RGB format)
            image_array = input_image.copy()
        else:
            image_array = np.array(input_image)
        
        # Ensure image is in the right format (uint8, 3 channels)
        if image_array.dtype != np.uint8:
            # Normalize if float (0-1 range)
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # Ensure 3 channels (RGB)
        if len(image_array.shape) == 2:
            # Grayscale to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
            # Unexpected number of channels
            raise ValueError(f"Unexpected image shape: {image_array.shape}")
        
        print(f"Processing image with shape: {image_array.shape}, dtype: {image_array.dtype}")
        
        # YOLO predict can handle RGB numpy arrays directly
        # Run inference (use CPU device)
        results = model.predict(
            image_array, 
            conf=confidence_threshold, 
            save=False, 
            verbose=False,
            device='cpu'
        )
        result = results[0]
        
        # Original image is already in RGB format (from Gradio)
        original_img = image_array.copy()
        
        # Get annotated image (result.plot() returns RGB numpy array)
        annotated_img = result.plot()
        
        # Ensure annotated image is numpy array
        if not isinstance(annotated_img, np.ndarray):
            annotated_img = np.array(annotated_img)
        
        # Create summary
        num_detections = len(result.boxes) if result.boxes is not None else 0
        summary_text = f"## Detection Results\n\n"
        summary_text += f"**Number of potholes detected:** {num_detections}\n\n"
        
        if num_detections > 0:
            summary_text += "**Detections:**\n"
            for i, box in enumerate(result.boxes, 1):
                conf = box.conf.item()
                summary_text += f"- Pothole {i}: {conf:.1%} confidence\n"
        else:
            summary_text += "No potholes detected in this image."
        
        return original_img, annotated_img, summary_text
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing image: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, f"**Error:** {str(e)}"

# Create simplified Gradio interface
def create_interface():
    """Create and launch the Gradio interface"""
    
    with gr.Blocks(title="Pothole Segmentation") as demo:
        gr.Markdown("# 🕳️ Pothole Detection & Segmentation")
        gr.Markdown("Upload an image to detect and segment potholes using YOLOv8")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Upload Image")
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold"
                )
                submit_btn = gr.Button("🔍 Detect Potholes", variant="primary")
            
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Original"):
                        original_output = gr.Image(label="Original Image")
                    with gr.Tab("Segmented"):
                        annotated_output = gr.Image(label="Segmented Image")
                    with gr.Tab("Side by Side"):
                        comparison_output = gr.Image(label="Comparison")
                summary_output = gr.Markdown()
        
        def process_image(img, conf):
            try:
                if img is None:
                    return None, None, None, "Please upload an image first."
                
                orig, annot, summary = segment_pothole(img, conf)
                
                # Create side-by-side comparison
                comparison = None
                if orig is not None and annot is not None:
                    if isinstance(orig, np.ndarray) and isinstance(annot, np.ndarray):
                        try:
                            # Ensure both images are the same height
                            h = max(orig.shape[0], annot.shape[0])
                            w1 = int(orig.shape[1] * h / orig.shape[0]) if orig.shape[0] > 0 else orig.shape[1]
                            w2 = int(annot.shape[1] * h / annot.shape[0]) if annot.shape[0] > 0 else annot.shape[1]
                            
                            orig_resized = cv2.resize(orig, (w1, h))
                            annot_resized = cv2.resize(annot, (w2, h))
                            comparison = np.hstack([orig_resized, annot_resized])
                        except Exception as e:
                            print(f"Error creating comparison: {e}")
                            comparison = None
                
                # Return results (Gradio can handle None, but let's ensure we return valid arrays)
                return orig, annot, comparison, summary
                
            except Exception as e:
                import traceback
                error_msg = f"Error in process_image: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                # Return error message in summary, keep images as None
                return None, None, None, f"**Error:** {str(e)}\n\nPlease check the terminal for details."
        
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[original_output, annotated_output, comparison_output, summary_output]
        )
        
        input_image.upload(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[original_output, annotated_output, comparison_output, summary_output]
        )
    
    return demo

if __name__ == "__main__":
    try:
        # Disable Gradio analytics to avoid connection timeout errors
        import os
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        
        demo = create_interface()
        # Try different launch configurations
        try:
            demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=False)
        except (ValueError, Exception) as e:
            print(f"Trying alternative launch method...")
            demo.launch(server_port=7860, share=False, show_error=False)
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you see a TypeError about 'bool' is not iterable,")
        print("this is a known Gradio bug. Try:")
        print("  pip3 install gradio==4.19.0 --force-reinstall")
        print("Or use the Jupyter notebook version instead.")
        sys.exit(1)

