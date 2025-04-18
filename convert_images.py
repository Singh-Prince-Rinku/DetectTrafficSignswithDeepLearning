#!/usr/bin/env python3
"""
Convert all .ppm images in the processed dataset to .jpg format.
"""

import os
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_images(base_dir):
    """Convert all .ppm images to .jpg in the given directory structure."""
    # Find all image directories
    image_dirs = []
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, split, 'images')
        if os.path.exists(img_dir):
            image_dirs.append(img_dir)
    
    # Process each directory
    for img_dir in image_dirs:
        print(f"Processing {img_dir}...")
        ppm_files = glob.glob(os.path.join(img_dir, '*.ppm'))
        
        for ppm_file in tqdm(ppm_files, desc=f"Converting images in {os.path.basename(os.path.dirname(img_dir))}"):
            try:
                # Open and convert image
                img = Image.open(ppm_file)
                
                # Get the path for the new jpg file
                jpg_file = os.path.splitext(ppm_file)[0] + '.jpg'
                
                # Save as JPG
                img.save(jpg_file, 'JPEG', quality=95)
                
                # Delete the original ppm file
                os.remove(ppm_file)
                
                # Update label file to reference jpg instead of ppm
                label_dir = os.path.join(os.path.dirname(os.path.dirname(ppm_file)), 'labels')
                label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(ppm_file))[0] + '.txt')
                
                # No need to change the label file content, as we're keeping the same filename base
                
            except Exception as e:
                print(f"Error processing {ppm_file}: {e}")

def main():
    # Base directory for processed data
    base_dir = 'data/processed'
    
    # Convert all images
    convert_images(base_dir)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 