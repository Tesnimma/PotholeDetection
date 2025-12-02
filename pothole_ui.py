"""
Pothole Segmentation Web UI
A beautiful Gradio interface for pothole detection and segmentation
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Try to import gradio, with helpful error message if not installed
try:
    import gradio as gr
except ImportError:
    print("Error: Gradio is not installed. Please install it with: pip3 install gradio")
    sys.exit(1)

# Load the trained model
def load_model():
    """Load the trained YOLO model"""
    model_paths = [
        'runs/segment/train/weights/best.pt',  # Training output directory
        'best.pt',  # Root directory
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
    
    Args:
        input_image: PIL Image or numpy array from Gradio
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        tuple: (original_image, annotated_image, summary_text)
    """
    if input_image is None:
        return None, None, "Please upload an image first."
    
    try:
        # Gradio passes numpy arrays in RGB format when type="numpy"
        # YOLO can handle numpy arrays directly
        if isinstance(input_image, Image.Image):
            image_array = np.array(input_image)
        elif isinstance(input_image, np.ndarray):
            image_array = input_image.copy()
        else:
            image_array = np.array(input_image)
        
        # Ensure image is in the right format (uint8)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Run inference (YOLO handles RGB numpy arrays directly)
        results = model.predict(image_array, conf=confidence_threshold, save=False, verbose=False)
        
        # Get the first result (single image)
        result = results[0]
        
        # Original image is already in RGB format
        original_img = image_array
        
        # Get annotated image with predictions
        annotated_img = result.plot()
        
        # Create summary text
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
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return None, None, error_msg

# Create Gradio interface
def create_interface():
    """Create and launch the Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Pothole Segmentation",
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🕳️ Pothole Detection & Segmentation
            ### Upload an image to detect and segment potholes using YOLOv8
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Upload Image")
                input_image = gr.Image(
                    type="numpy",
                    label="Upload Image",
                    height=400
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold (adjust detection sensitivity)"
                )
                
                submit_btn = gr.Button(
                    "🔍 Detect Potholes",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📊 Results")
                
                with gr.Tabs():
                    with gr.Tab("Original Image"):
                        original_output = gr.Image(
                            label="Original Image",
                            height=400
                        )
                    
                    with gr.Tab("Segmented Image"):
                        annotated_output = gr.Image(
                            label="Segmented Image with Detections",
                            height=400
                        )
                    
                    with gr.Tab("Side by Side"):
                        comparison_output = gr.Image(
                            label="Original vs Segmented",
                            height=400
                        )
                
                summary_output = gr.Markdown(
                    value="Upload an image and click 'Detect Potholes' to see results."
                )
        
        # Examples section
        gr.Markdown("### 📷 Example Images")
        example_images = []
        valid_images_dir = 'Pothole_Segmentation_YOLOv8/valid/images'
        if os.path.exists(valid_images_dir):
            valid_images = [f for f in os.listdir(valid_images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if valid_images:
                # Use first few images as examples
                example_images = [os.path.join(valid_images_dir, img) 
                                for img in valid_images[:6]]
        
        if example_images:
            gr.Markdown("**Click on an example image below to test:**")
            gr.Examples(
                examples=example_images,
                inputs=input_image
            )
        
        # Process function that handles side-by-side view
        def process_image(img, conf):
            orig, annot, summary = segment_pothole(img, conf)
            
            # Create side-by-side comparison
            if orig is not None and annot is not None:
                # Resize images to same height for comparison
                h = max(orig.shape[0], annot.shape[0])
                w1 = int(orig.shape[1] * h / orig.shape[0])
                w2 = int(annot.shape[1] * h / annot.shape[0])
                
                orig_resized = cv2.resize(orig, (w1, h))
                annot_resized = cv2.resize(annot, (w2, h))
                
                # Create side-by-side image
                comparison = np.hstack([orig_resized, annot_resized])
            else:
                comparison = None
            
            return orig, annot, comparison, summary
        
        # Connect components
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[original_output, annotated_output, comparison_output, summary_output]
        )
        
        # Auto-process when image is uploaded
        input_image.upload(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[original_output, annotated_output, comparison_output, summary_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **Note:** This model uses YOLOv8 for pothole detection and segmentation. 
            Adjust the confidence threshold to control detection sensitivity.
            """
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    try:
        demo = create_interface()
        # Try launching with localhost first
        try:
            demo.launch(
                share=False,
                server_name="127.0.0.1",
                server_port=7860,
                show_error=True,
                inbrowser=False
            )
        except ValueError as e:
            # If localhost fails, try with share=True or different server_name
            print(f"Note: {e}")
            print("Trying alternative launch configuration...")
            demo.launch(
                share=False,
                server_name=None,  # Let Gradio choose
                server_port=7860,
                show_error=True,
                inbrowser=False
            )
    except Exception as e:
        print(f"Error launching interface: {e}")
        print("\nTroubleshooting tips:")
        print("1. Try upgrading Gradio: pip3 install --upgrade gradio")
        print("2. Or try a specific version: pip3 install gradio==4.19.0")
        print("3. Check if port 7860 is available")
        sys.exit(1)
# To run this script:
# pip install gradio
# python3 pothole_ui.py
