#!/usr/bin/env python3
"""
Train a YOLOv8 model on the GTSRB dataset using the simplified API.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for traffic sign detection')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--model-size', type=str, default='s', help='YOLOv8 model size (n, s, m, l, x)')
    args = parser.parse_args()
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Path to the data.yaml file
    data_yaml_path = 'data/processed/data.yaml'
    
    # Load a pre-trained YOLOv8 model
    model_path = f"yolov8{args.model_size}.pt"
    model = YOLO(model_path)
    
    # Print training configuration
    print("Training configuration:")
    print(f"  Model: YOLOv8{args.model_size}")
    print(f"  Dataset: {data_yaml_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print()
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project='traffic_sign_detection',
        name=f'yolov8{args.model_size}_gtsrb'
    )
    
    # Save the model in 'models' directory
    model_save_path = Path('models') / f'best_yolov8{args.model_size}_gtsrb.pt'
    best_model_path = Path(f'traffic_sign_detection/yolov8{args.model_size}_gtsrb/weights/best.pt')
    
    if best_model_path.exists():
        os.system(f'cp {best_model_path} {model_save_path}')
        print(f"Best model saved to {model_save_path}")
    
    print("Training completed!")
    print("Next steps:")
    print("1. Evaluate the model using evaluate.py")
    print("2. Run inference using detect.py")

if __name__ == "__main__":
    main() 