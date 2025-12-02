# Pothole Detection & Segmentation using YOLOv8

A complete project for detecting and segmenting potholes in images using YOLOv8 segmentation model. This project includes training scripts, inference notebooks, and a web interface for easy image analysis.

## 📋 Table of Contents

- [Quick Start](#-quick-start-3-steps)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Web Interface (Recommended)](#option-1-web-interface-recommended)
  - [Jupyter Notebooks](#option-2-jupyter-notebooks)
  - [Command Line](#option-3-command-line-advanced)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

Open a terminal/command prompt in this folder and run:

```bash
pip3 install -r requirements.txt
```

**Windows users:** Use `pip` instead of `pip3`  
**macOS/Linux users:** Use `pip3` if `pip` points to Python 2

### Step 2: Verify Setup (Optional but Recommended)

Check if everything is installed correctly:

```bash
python3 verify_setup.py
```

This will verify:
- ✅ Python version
- ✅ All required packages
- ✅ Model files
- ✅ Dataset files

### Step 3: Launch the Web Interface

```bash
streamlit run pothole_streamlit.py
```

**Windows users:** If `streamlit` command doesn't work:
```bash
python -m streamlit run pothole_streamlit.py
```

**macOS/Linux users:** If `streamlit` command doesn't work:
```bash
python3 -m streamlit run pothole_streamlit.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 4: Use the Interface

1. **Upload an Image**: Click "Browse files" or drag and drop an image (JPG, PNG, BMP)
2. **Adjust Confidence** (optional): Use the slider in the sidebar (default: 0.25)
   - Lower values (0.1-0.3) = more detections (may include false positives)
   - Higher values (0.5-1.0) = fewer detections (only high-confidence)
3. **Detect Potholes**: Click the "🔍 Detect Potholes" button
4. **View Results**: 
   - **Original Tab**: The uploaded image
   - **Segmented Tab**: Image with pothole detections and segmentation masks
   - **Side by Side Tab**: Comparison view of original and segmented images
   - **Detection Summary**: Table showing number of potholes and confidence scores

## ✨ Features

- 🎯 **YOLOv8 Segmentation Model**: State-of-the-art pothole detection and segmentation
- 🖼️ **Web Interface**: Easy-to-use Streamlit web app for image upload and analysis
- 📊 **Multiple Views**: Original, segmented, and side-by-side comparison views
- ⚙️ **Adjustable Confidence**: Control detection sensitivity with a slider
- 📈 **Detailed Results**: Detection count and confidence scores for each pothole
- 📓 **Jupyter Notebooks**: Interactive notebooks for training and inference

## 🔧 Prerequisites

- **Python 3.9 or higher** (Python 3.9.6 recommended)
- **pip** (Python package manager)
- **Web browser** (for Streamlit interface)
- **Jupyter Notebook** (optional, only if using notebooks)

## 📦 Installation

### Step 1: Extract/Download the Project

If you received this as a zip file, extract it to a folder. If using git:

```bash
git clone <repository-url>
cd pothole
```

### Step 2: Install Python Dependencies

Install all required packages:

```bash
pip3 install -r requirements.txt
```

**Platform-specific notes:**
- **Windows:** Use `pip` instead of `pip3`
- **macOS/Linux:** Use `pip3` to ensure Python 3 is used

### Step 3: Verify Installation

Run the verification script:

```bash
python3 verify_setup.py
```

Or manually verify:

```bash
python3 -c "import ultralytics, streamlit, cv2; print('✅ All packages installed successfully!')"
```

## 📁 Project Structure

```
pothole/
├── README.md                          # This file - start here!
├── requirements.txt                   # Python dependencies
├── verify_setup.py                   # Setup verification script
│
├── best.pt                            # ✅ Trained model (REQUIRED for inference)
├── yolov8s-seg.pt                    # Pre-trained YOLOv8 base model
│
├── pothole_streamlit.py              # ✅ Web interface (RECOMMENDED)
├── pothole-segmentation-using-yolov8.ipynb  # Training notebook
├── pothole-inference.ipynb            # Inference notebook
│
└── Pothole_Segmentation_YOLOv8/      # Dataset directory
    ├── data.yaml                     # Dataset configuration
    ├── train/                        # Training images and labels (720 images)
    │   ├── images/
    │   └── labels/
    └── valid/                        # Validation images and labels (60 images)
        ├── images/
        └── labels/
```

## 🚀 Usage

### Option 1: Web Interface (Recommended)

The easiest and most user-friendly way to use the project.

#### Start the Web Server

```bash
streamlit run pothole_streamlit.py
```

#### Access the Interface

- The app will automatically open in your browser
- If not, manually navigate to: `http://localhost:8501`
- For network access: `http://0.0.0.0:8501`

#### Using the Interface

1. **Upload Image**: Drag and drop or click to upload (JPG, JPEG, PNG, BMP)
2. **Adjust Settings**: Use sidebar slider to adjust confidence threshold
3. **Process**: Click "🔍 Detect Potholes" button
4. **View Results**: Switch between tabs to see:
   - Original image
   - Segmented image with detections
   - Side-by-side comparison
   - Detection summary table

### Option 2: Jupyter Notebooks

#### For Inference (Testing the Model)

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open** `pothole-inference.ipynb`

3. **Run cells in order**:
   - Cell 1: Install/import libraries
   - Cell 2: Load the model
   - Cell 3: Define segmentation function
   - Cell 4-9: Various inference methods (file upload, file path, validation images)

#### For Training (Optional - Only if you want to retrain)

**Note:** The pre-trained model (`best.pt`) is included and ready to use. Training is only needed if you want to improve or modify the model.

1. **Open** `pothole-segmentation-using-yolov8.ipynb`

2. **Run cells in order**:
   - Cell 0: Install dependencies
   - Cell 1-2: Setup and load base model
   - Cell 3-6: Explore dataset
   - Cell 7: **Train the model** ⚠️ (This takes significant time - hours on CPU!)
   - Cell 8-11: Evaluate and visualize results

**Training Requirements:**
- Significant time (several hours on CPU, less on GPU)
- Computational resources
- The trained model will be saved to `runs/segment/train/weights/best.pt`

### Option 3: Command Line (Advanced)

You can use the model directly in Python:

```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference on an image
results = model.predict('path/to/your/image.jpg', conf=0.25)

# View results
results[0].show()

# Get detection count
num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
print(f"Detected {num_detections} potholes")
```

## 📦 Model Files

The project expects the trained model in one of these locations (checked in order):

1. `runs/segment/train/weights/best.pt` (after training)
2. `best.pt` (in the root directory) ✅ **This file is included**

The `best.pt` file in the root directory is the trained model and works immediately for inference. No training required!

## 🐛 Troubleshooting

### Issue: "Model file not found"

**Solution:** 
- Ensure `best.pt` exists in the project root directory
- If you trained the model, it should be in `runs/segment/train/weights/best.pt`
- Run `python3 verify_setup.py` to check

### Issue: "pip: command not found" or "python: command not found"

**Solution:** 
- **macOS/Linux:** Use `pip3` and `python3` instead
- **Windows:** Ensure Python is added to PATH during installation
- Check Python installation: `python3 --version` (should be 3.9+)

### Issue: "streamlit: command not found"

**Solution:** 
```bash
python3 -m streamlit run pothole_streamlit.py
```

Or reinstall Streamlit:
```bash
pip3 install streamlit
```

### Issue: Port 8501 already in use

**Solution:** Use a different port:
```bash
streamlit run pothole_streamlit.py --server.port 8502
```

### Issue: Import errors or missing packages

**Solution:** 
1. Reinstall requirements:
   ```bash
   pip3 install -r requirements.txt --upgrade
   ```
2. Verify installation:
   ```bash
   python3 verify_setup.py
   ```

### Issue: CUDA/GPU errors

**Solution:** 
- The code is set to use CPU by default (works on all systems)
- If you have a GPU and want to use it:
  - Install PyTorch with CUDA support
  - Modify `device='cpu'` to `device=0` in the code

### Issue: Slow processing

**Solution:** 
- Processing on CPU is slower than GPU (normal)
- Large images take longer to process
- Typical processing time: 1-5 seconds per image on CPU
- Consider resizing very large images before uploading

### Issue: "Permission denied" or installation errors

**Solution:**
- **macOS/Linux:** Try with `--user` flag:
  ```bash
  pip3 install -r requirements.txt --user
  ```
- **Windows:** Run command prompt as Administrator
- Use virtual environment (recommended):
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## 📝 Important Notes

- **First Run**: The first time you run inference, the model may take a moment to load
- **Image Formats**: Supported formats are JPG, JPEG, PNG, and BMP
- **Confidence Threshold**: 
  - Lower values (0.1-0.3) = more detections (may include false positives)
  - Higher values (0.5-1.0) = fewer detections (only high-confidence)
  - Default: 0.25 (good balance)
- **Processing Time**: Typically 1-5 seconds per image on CPU, depending on image size
- **Model File**: The `best.pt` file (~22 MB) must be in the project directory

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
- **Dataset Info**: See `Pothole_Segmentation_YOLOv8/README.roboflow.txt`

## 🤝 Getting Help

If you encounter any issues:

1. **Run the verification script:**
   ```bash
   python3 verify_setup.py
   ```

2. **Check the Troubleshooting section** above

3. **Verify dependencies:**
   ```bash
   pip3 list | grep -E "(ultralytics|streamlit|opencv)"
   ```

4. **Check Python version:**
   ```bash
   python3 --version
   ```
   Should be 3.9 or higher

5. **Ensure model file exists:**
   ```bash
   ls -la best.pt
   ```

## 📄 License

This project uses a dataset licensed under CC BY 4.0. See `Pothole_Segmentation_YOLOv8/README.roboflow.txt` for details.

---

## 🎯 Summary

**To get started right now:**

1. `pip3 install -r requirements.txt`
2. `streamlit run pothole_streamlit.py`
3. Upload an image and click "Detect Potholes"

**That's it!** 🚀
