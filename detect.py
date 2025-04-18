#!/usr/bin/env python3
"""
Run inference with a trained YOLOv8 model on images, videos, or webcam feed.
"""

import os
import sys
import yaml
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

def load_data_yaml(data_yaml_path):
    """Load data.yaml file and return class names."""
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    class_names = {}
    for k, v in data_yaml.get('names', {}).items():
        class_names[int(k)] = v
    
    return class_names

def draw_detections(image, results, class_names, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Image to draw on
        results: YOLOv8 prediction results
        class_names: Dictionary mapping class IDs to class names
        conf_threshold: Minimum confidence threshold for displaying detections
    
    Returns:
        Image with detections drawn
    """
    # Create a copy of the image
    annotated_img = image.copy()
    
    # Get the image height and width
    h, w = annotated_img.shape[:2]
    
    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Skip low confidence detections
            if conf < conf_threshold:
                continue
            
            # Get class name
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            
            # Generate random color based on class id
            color = get_color(cls_id)
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{cls_name} {conf:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_img, 
                (x1, y1 - label_height - baseline - 5), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_img, 
                label, 
                (x1, y1 - baseline - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
    
    return annotated_img

def get_color(class_id):
    """Generate a color based on class id."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64)
    ]
    
    return colors[class_id % len(colors)]

def process_image(model, image_path, output_dir, class_names, conf_threshold=0.25, image_size=640):
    """
    Process a single image with the model.
    
    Args:
        model: YOLOv8 model
        image_path: Path to the image
        output_dir: Directory to save the output
        class_names: Dictionary mapping class IDs to class names
        conf_threshold: Minimum confidence threshold for displaying detections
        image_size: Input image size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Run inference
    results = model.predict(
        source=image,
        imgsz=image_size,
        conf=conf_threshold,
        verbose=False
    )
    
    # Draw detections
    annotated_img = draw_detections(image, results, class_names, conf_threshold)
    
    # Save the annotated image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, annotated_img)
    
    print(f"Processed image saved to {output_path}")
    
    return annotated_img, results

def process_video(model, video_path, output_dir, class_names, conf_threshold=0.25, image_size=640):
    """
    Process a video with the model.
    
    Args:
        model: YOLOv8 model
        video_path: Path to the video (or 0 for webcam)
        output_dir: Directory to save the output
        class_names: Dictionary mapping class IDs to class names
        conf_threshold: Minimum confidence threshold for displaying detections
        image_size: Input image size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    if video_path == "0" or video_path == 0:
        cap = cv2.VideoCapture(0)
        output_path = os.path.join(output_dir, "webcam_output.mp4")
    else:
        cap = cv2.VideoCapture(str(video_path))
        output_path = os.path.join(output_dir, os.path.basename(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_path != 0 else float('inf')
    
    # For FPS calculation
    start_time = time.time()
    frames_processed = 0
    
    # Create a progress bar for file videos
    if video_path != 0 and video_path != "0":
        pbar = tqdm(total=total_frames, desc="Processing video")
    
    try:
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = model.predict(
                source=frame,
                imgsz=image_size,
                conf=conf_threshold,
                verbose=False
            )
            
            # Draw detections
            annotated_frame = draw_detections(frame, results, class_names, conf_threshold)
            
            # Calculate and display FPS
            frames_processed += 1
            elapsed_time = time.time() - start_time
            fps_current = frames_processed / elapsed_time if elapsed_time > 0 else 0
            
            # Add FPS text
            cv2.putText(
                annotated_frame,
                f"FPS: {fps_current:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Write the frame
            out.write(annotated_frame)
            
            # Display the frame (for webcam)
            if video_path == 0 or video_path == "0":
                cv2.imshow("Traffic Sign Detection", annotated_frame)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Update progress bar
                pbar.update(1)
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if video_path != 0 and video_path != "0":
            pbar.close()
        
        print(f"Processed video saved to {output_path}")
        print(f"Processed {frame_count} frames at {fps_current:.1f} FPS")

def main():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv8 traffic sign detection model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image/video file or 0 for webcam')
    parser.add_argument('--weights', type=str, default='models/best_yolov8s_gtsrb.pt',
                       help='Path to model weights')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save output')
    args = parser.parse_args()
    
    # Load the model
    model = YOLO(args.weights)
    
    # Load class names
    data_yaml_path = Path('data/processed/data.yaml')
    if data_yaml_path.exists():
        class_names = load_data_yaml(data_yaml_path)
    else:
        # Fallback to model class names
        class_names = {i: name for i, name in enumerate(model.names)}
    
    # Check if source is an image, video, or webcam
    source_path = args.source
    
    if source_path == "0":
        # Webcam
        print("Running inference on webcam feed...")
        process_video(
            model,
            0,
            args.output_dir,
            class_names,
            args.conf_thresh,
            args.img_size
        )
    elif os.path.isfile(source_path):
        # Check file extension
        ext = os.path.splitext(source_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Image
            print(f"Running inference on image: {source_path}")
            process_image(
                model,
                source_path,
                args.output_dir,
                class_names,
                args.conf_thresh,
                args.img_size
            )
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            # Video
            print(f"Running inference on video: {source_path}")
            process_video(
                model,
                source_path,
                args.output_dir,
                class_names,
                args.conf_thresh,
                args.img_size
            )
        else:
            print(f"Unsupported file format: {ext}")
    else:
        print(f"Error: Source {source_path} not found")

if __name__ == "__main__":
    main() 