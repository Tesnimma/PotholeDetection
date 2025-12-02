#!/usr/bin/env python3
"""
Setup Verification Script
Run this to check if all dependencies are installed correctly
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False

def check_file(filepath):
    """Check if a file exists"""
    import os
    if os.path.exists(filepath):
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False

def main():
    print("=" * 60)
    print("Pothole Detection Project - Setup Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print("Python Version:")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ⚠️  Warning: Python 3.9+ recommended")
    else:
        print("  ✅ Python version OK")
    print()
    
    # Check required packages
    print("Required Packages:")
    packages = [
        ("ultralytics", "ultralytics"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("streamlit", "streamlit"),
        ("pyyaml", "yaml"),
    ]
    
    all_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    print()
    
    # Check optional packages
    print("Optional Packages:")
    optional_packages = [
        ("jupyter", "jupyter"),
        ("seaborn", "seaborn"),
    ]
    
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)
    
    print()
    
    # Check model files
    print("Model Files:")
    import os
    model_paths = [
        'best.pt',
        'runs/segment/train/weights/best.pt',
    ]
    
    model_found = False
    for path in model_paths:
        if check_file(path):
            model_found = True
            break
    
    if not model_found:
        print("  ⚠️  Warning: No model file found. Training required.")
    
    print()
    
    # Check dataset
    print("Dataset:")
    dataset_paths = [
        'Pothole_Segmentation_YOLOv8/data.yaml',
        'Pothole_Segmentation_YOLOv8/train/images',
        'Pothole_Segmentation_YOLOv8/valid/images',
    ]
    
    for path in dataset_paths:
        check_file(path)
    
    print()
    print("=" * 60)
    
    if all_ok and model_found:
        print("✅ Setup looks good! You're ready to go!")
        print()
        print("Next steps:")
        print("  1. Run: streamlit run pothole_streamlit.py")
        print("  2. Or open: pothole-inference.ipynb in Jupyter")
    elif all_ok:
        print("⚠️  Packages installed, but model file missing.")
        print("   You can still train the model using the training notebook.")
    else:
        print("❌ Some packages are missing.")
        print("   Run: pip3 install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

