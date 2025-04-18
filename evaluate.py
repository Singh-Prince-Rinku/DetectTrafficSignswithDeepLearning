#!/usr/bin/env python3
"""
Evaluate a trained YOLOv8 model on the GTSRB test dataset.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

def load_data_yaml(data_yaml_path):
    """Load data.yaml file and return class names."""
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    class_names = {}
    for k, v in data_yaml.get('names', {}).items():
        class_names[int(k)] = v
    
    return class_names

def evaluate_model(model_path, data_path=None, conf_thresh=0.25, iou_thresh=0.5, image_size=640):
    """
    Evaluate the YOLOv8 model on the test dataset.
    
    Args:
        model_path: Path to the trained model weights
        data_path: Path to the test data directory (if None, use from data.yaml)
        conf_thresh: Confidence threshold for detections
        iou_thresh: IoU threshold for NMS
        image_size: Input image size
        
    Returns:
        Results dictionary
    """
    # Load the model
    model = YOLO(model_path)
    
    # Path to data.yaml
    data_yaml_path = Path('data/processed/data.yaml')
    if not data_yaml_path.exists():
        print(f"Error: data.yaml not found at {data_yaml_path}")
        print("Please specify the test data path or run the preprocessing script.")
        sys.exit(1)
    
    # Get class names
    class_names = load_data_yaml(data_yaml_path)
    
    # Run validation - Use the data.yaml file directly instead of the test directory path
    print(f"Evaluating model using {data_yaml_path}...")
    results = model.val(
        data=str(data_yaml_path),
        imgsz=image_size,
        conf=conf_thresh,
        iou=iou_thresh,
        save_json=True,
        save_hybrid=True,
        verbose=True
    )
    
    # Print results summary
    print("\nEvaluation Results:")
    print(f"mAP@{iou_thresh:.2f}:      {results.box.map:.4f}")
    print(f"mAP@0.5:0.95:    {results.box.map50_95:.4f}")
    print(f"Precision:       {results.box.p:.4f}")
    print(f"Recall:          {results.box.r:.4f}")
    print(f"F1-Score:        {results.box.f1:.4f}")
    print(f"Speed (ms/img):  {results.speed['inference']:.1f}")
    
    # Print per-class results
    print("\nPer-class Results:")
    df = pd.DataFrame({
        'Class': [class_names.get(i, f"Class {i}") for i in range(len(results.names))],
        'AP@0.5': results.box.ap50.tolist(),
        'Precision': results.box.p_curve[:, 0, -1].tolist(), 
        'Recall': results.box.r_curve[:, 0, -1].tolist()
    })
    
    # Sort by AP50 and print top and bottom 5 classes
    df_sorted = df.sort_values('AP@0.5', ascending=False)
    
    print("\nTop 5 Classes:")
    print(df_sorted.head(5).to_string(index=False))
    
    print("\nBottom 5 Classes:")
    print(df_sorted.tail(5).to_string(index=False))
    
    # Create confusion matrix plot
    plot_confusion_matrix(results, class_names)
    
    return results

def plot_confusion_matrix(results, class_names, output_path='evaluation'):
    """Plot and save the confusion matrix."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            cm = results.confusion_matrix.matrix
            
            # Normalize the confusion matrix
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
            
            # Plot the confusion matrix
            plt.figure(figsize=(12, 10))
            plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Normalized Confusion Matrix')
            plt.colorbar()
            
            # Reduce number of classes if too many
            if len(class_names) > 20:
                # Plot only every 5th tick
                tick_marks = np.arange(0, len(class_names), 5)
                tick_labels = [class_names.get(i, f"Class {i}") for i in range(0, len(class_names), 5)]
            else:
                tick_marks = np.arange(len(class_names))
                tick_labels = [class_names.get(i, f"Class {i}") for i in range(len(class_names))]
            
            plt.xticks(tick_marks, tick_labels, rotation=90)
            plt.yticks(tick_marks, tick_labels)
            
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save the confusion matrix
            plt.savefig(f"{output_path}/confusion_matrix.png", dpi=200, bbox_inches='tight')
            print(f"Confusion matrix saved to {output_path}/confusion_matrix.png")
        else:
            print("Confusion matrix not available in results.")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 traffic sign detection model')
    parser.add_argument('--weights', type=str, default='models/best_yolov8s_gtsrb.pt',
                       help='Path to model weights')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to test data directory')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    args = parser.parse_args()
    
    results = evaluate_model(
        args.weights, 
        args.data, 
        args.conf_thresh, 
        args.iou_thresh, 
        args.img_size
    )
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main() 