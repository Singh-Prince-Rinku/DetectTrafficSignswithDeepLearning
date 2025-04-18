#!/usr/bin/env python3
"""
Train a YOLOv8 model on the GTSRB dataset for traffic sign detection.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config_path):
    """Train YOLOv8 model with the given configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data configuration - Fixed path to data.yaml
    data_yaml_path = Path('data/processed/data.yaml')
    if not data_yaml_path.exists():
        print(f"Error: data.yaml not found at {data_yaml_path}")
        print("Please run the preprocessing script first.")
        sys.exit(1)
    
    # Determine model size (nano, small, medium, large, etc.)
    model_size = config['model']['size']
    model_path = f"yolov8{model_size}.pt"
    
    # Initialize model with pre-trained weights
    model = YOLO(model_path)
    
    # Prepare training arguments
    train_args = {
        'data': str(data_yaml_path),
        'epochs': config['training']['epochs'],
        'patience': config['training']['patience'],
        'batch': config['dataset']['batch_size'],
        'imgsz': config['dataset']['img_size'],
        'lr0': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay'],
        'optimizer': config['training']['optimizer'].lower(),
        'workers': config['dataset']['num_workers'],
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'project': 'traffic_sign_detection',
        'name': f'yolov8{model_size}_gtsrb',
        'pretrained': config['model']['pretrained'],
        'cos_lr': config['training']['lr_scheduler'] == 'cosine',
        'warmup_epochs': config['training']['warmup_epochs'],
        'hsv_h': config['augmentation']['hsv_h'],
        'hsv_s': config['augmentation']['hsv_s'],
        'hsv_v': config['augmentation']['hsv_v'],
        'degrees': config['augmentation']['rotate'],
        'translate': config['augmentation']['translate'],
        'scale': config['augmentation']['scale'],
        'fliplr': config['augmentation']['fliplr'],
        'flipud': config['augmentation']['flipud'],
        'mosaic': config['augmentation']['mosaic'],
        'mixup': config['augmentation']['mixup'],
        'box': config['loss']['box'],
        'cls': config['loss']['cls'],
        'obj': config['loss']['obj'],
        'save_period': config['save']['save_period'],
        'save': config['save']['save_best'],  # save best model
    }
    
    # Print training configuration
    print("Training configuration:")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Dataset: {data_yaml_path}")
    print(f"  Device: {train_args['device']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['dataset']['batch_size']}")
    print(f"  Image size: {config['dataset']['img_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print()
    
    # Train the model
    print("Starting training...")
    model.train(**train_args)
    
    # Save the model in 'models' directory
    os.makedirs('models', exist_ok=True)
    model_save_path = Path('models') / f'best_yolov8{model_size}_gtsrb.pt'
    if Path(f'traffic_sign_detection/yolov8{model_size}_gtsrb/weights/best.pt').exists():
        os.system(f'cp traffic_sign_detection/yolov8{model_size}_gtsrb/weights/best.pt {model_save_path}')
        print(f"Best model saved to {model_save_path}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for traffic sign detection')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to training configuration file')
    args = parser.parse_args()
    
    model = train_model(args.config)
    
    print("Training completed!")
    print("Next steps:")
    print("1. Evaluate the model using evaluate.py")
    print("2. Run inference using detect.py")

if __name__ == "__main__":
    main() 