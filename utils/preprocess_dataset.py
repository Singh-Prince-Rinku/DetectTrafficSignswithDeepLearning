#!/usr/bin/env python3
"""
Preprocess the GTSRB dataset and convert it to YOLOv8 format.

The GTSRB dataset comes with ROI (Region of Interest) annotations,
but we need to convert them to the YOLO format (class_id, x_center, y_center, width, height).
"""

import os
import csv
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import random
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def parse_annotations_from_csv(csv_file, img_dir, one_indexed=False):
    """
    Parse annotations from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        img_dir: Directory containing the images
        one_indexed: Whether the class IDs in the CSV are 1-indexed (should be converted to 0-indexed)
        
    Returns:
        Dictionary mapping image paths to their annotations: [class_id, x1, y1, x2, y2]
    """
    annotations = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header
        
        for row in reader:
            try:
                if len(row) < 7:  # Check if row has enough columns
                    continue
                    
                img_filename = row[0]
                width = int(row[1])
                height = int(row[2])
                class_id = int(row[7])
                
                # Convert to 0-indexed if necessary
                if one_indexed:
                    class_id -= 1
                    
                # Get ROI coordinates
                x1 = int(row[3])
                y1 = int(row[4])
                x2 = int(row[5])
                y2 = int(row[6])
                
                img_path = os.path.join(img_dir, img_filename)
                if img_path not in annotations:
                    annotations[img_path] = []
                    
                annotations[img_path].append([class_id, x1, y1, x2, y2])
            except (IndexError, ValueError) as e:
                print(f"Error parsing row {row}: {e}")
                
    return annotations

def convert_to_yolo_format(annotations, img_width, img_height):
    """
    Convert [class_id, x1, y1, x2, y2] to YOLO format [class_id, x_center, y_center, width, height].
    All values are normalized to [0, 1].
    """
    class_id, x1, y1, x2, y2 = annotations
    
    # Calculate normalized center coordinates and dimensions
    x_center = (x1 + x2) / (2 * img_width)
    y_center = (y1 + y2) / (2 * img_height)
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return [class_id, x_center, y_center, width, height]

def process_training_data(raw_dir, processed_dir, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Process the GTSRB training dataset.
    
    The GTSRB training data is organized by class in separate directories.
    Each class directory contains images and a GT-<class_id>.csv file with annotations.
    """
    # Create output directories
    train_img_dir = processed_dir / "train" / "images"
    val_img_dir = processed_dir / "val" / "images"
    test_img_dir = processed_dir / "test" / "images"
    
    train_label_dir = processed_dir / "train" / "labels"
    val_label_dir = processed_dir / "val" / "labels"
    test_label_dir = processed_dir / "test" / "labels"
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Process each class directory
    class_dirs = sorted(glob.glob(str(raw_dir / "train" / "GTSRB" / "Final_Training" / "Images" / "*")))
    
    all_images = []
    all_annotations = {}
    
    for class_dir in tqdm(class_dirs, desc="Processing training classes"):
        class_id = int(os.path.basename(class_dir))
        
        # Read annotations
        csv_file = glob.glob(os.path.join(class_dir, "GT-*.csv"))[0]
        img_dir = class_dir
        
        annotations = parse_annotations_from_csv(csv_file, img_dir)
        
        for img_path, annots in annotations.items():
            if not os.path.exists(img_path):
                continue
                
            img_filename = os.path.basename(img_path)
            all_images.append(img_path)
            all_annotations[img_path] = annots
    
    # Split data into train, validation, test sets
    random.seed(42)  # For reproducibility
    all_images = sorted(all_images)  # Sort for reproducibility
    random.shuffle(all_images)
    
    n_samples = len(all_images)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    # Process train, validation, test sets
    for img_set, img_dir, label_dir in [
        (train_images, train_img_dir, train_label_dir),
        (val_images, val_img_dir, val_label_dir),
        (test_images, test_img_dir, test_label_dir)
    ]:
        for img_path in tqdm(img_set, desc=f"Processing {os.path.basename(img_dir.parent)} set"):
            try:
                # Copy the image
                img_filename = os.path.basename(img_path)
                dest_img_path = os.path.join(img_dir, img_filename)
                shutil.copy(img_path, dest_img_path)
                
                # Create YOLO format annotation
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_path = os.path.join(label_dir, label_filename)
                
                with open(label_path, 'w') as f:
                    for annot in all_annotations[img_path]:
                        yolo_annot = convert_to_yolo_format(annot, img_width, img_height)
                        f.write(" ".join(map(str, yolo_annot)) + "\n")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def create_data_yaml(processed_dir, num_classes=43):
    """Create the data.yaml file required by YOLOv8."""
    yaml_content = f"""# GTSRB dataset for YOLOv8
train: {processed_dir / 'train'}
val: {processed_dir / 'val'}
test: {processed_dir / 'test'}

# Number of classes
nc: {num_classes}

# Class names
names:
"""

    # Define class names for the GTSRB dataset
    class_names = [
        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
        "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
        "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
        "No passing", "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
        "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
        "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
        "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
        "Road work", "Traffic signals", "Pedestrians", "Children crossing",
        "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
        "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
        "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
        "Keep left", "Roundabout mandatory", "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons"
    ]

    # Add class names to YAML
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: '{name}'\n"

    # Write YAML file
    with open(processed_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    parser = argparse.ArgumentParser(description='Preprocess GTSRB dataset for YOLOv8')
    parser.add_argument('--data-dir', type=str, default='data', 
                        help='Path to data directory')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Proportion of data for training')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Proportion of data for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Proportion of data for testing')
    args = parser.parse_args()
    
    # Update paths based on args
    global DATA_DIR, RAW_DIR, PROCESSED_DIR
    DATA_DIR = Path(args.data_dir)
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Process the training data
    print("Processing training data...")
    process_training_data(
        RAW_DIR, 
        PROCESSED_DIR,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Create data.yaml file
    print("Creating data.yaml file...")
    create_data_yaml(PROCESSED_DIR)
    
    print("Preprocessing completed successfully!")
    print(f"Processed data is available at: {PROCESSED_DIR}")
    print("Next step: Train the model using the training script")

if __name__ == "__main__":
    main() 