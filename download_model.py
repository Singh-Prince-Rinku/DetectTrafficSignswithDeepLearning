#!/usr/bin/env python3
"""
Download YOLOv8 pre-trained model weights.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """Download YOLOv8 model weights."""
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Model size to download (nano, small, medium, large, etc.)
    model_size = 's'  # We're using the small version as specified in the config
    
    print(f"Downloading YOLOv8{model_size} model weights...")
    
    # Initialize model with pre-trained weights (this will download them)
    model = YOLO(f"yolov8{model_size}.pt")
    
    # Save a copy to the models directory
    model_path = Path('models') / f"yolov8{model_size}.pt"
    model.save(model_path)
    
    print(f"Model weights saved to {model_path}")
    print("The model is ready for training.")

if __name__ == "__main__":
    main() 