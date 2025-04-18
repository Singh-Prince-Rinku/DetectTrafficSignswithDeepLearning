#!/usr/bin/env python3
"""
Simple script to download and extract the GTSRB dataset with SSL verification disabled (for testing).
"""

import os
import ssl
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm

# URLs for the GTSRB dataset
GTSRB_TRAIN_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
GTSRB_TEST_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
GTSRB_TEST_GT_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from url to output_path with SSL verification disabled."""
    # Create an SSL context that doesn't verify certificates
    ssl_context = ssl._create_unverified_context()
    
    # Create a request
    req = urllib.request.Request(url)
    
    try:
        # Open the URL with the SSL context
        with urllib.request.urlopen(req, context=ssl_context) as response:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 8192
            
            # Setup progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                # Open output file for writing
                with open(output_path, 'wb') as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                            
                        # Write the data to the file
                        out_file.write(buffer)
                        
                        # Update the progress bar
                        pbar.update(len(buffer))
                        
    except urllib.error.URLError as e:
        print(f"Error downloading {url}: {e}")
        raise

def extract_zip(zip_path, extract_dir):
    """Extract a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f'Extracting {zip_path}'):
            try:
                zip_ref.extract(member, extract_dir)
            except zipfile.error as e:
                print(f"Error extracting {member.filename}: {e}")

def main():
    # Create directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Download training data
    train_zip = RAW_DIR / "GTSRB_Final_Training_Images.zip"
    print(f"Downloading training data from {GTSRB_TRAIN_URL}")
    download_url(GTSRB_TRAIN_URL, train_zip)
    
    # Download test data
    test_zip = RAW_DIR / "GTSRB_Final_Test_Images.zip"
    print(f"Downloading test data from {GTSRB_TEST_URL}")
    download_url(GTSRB_TEST_URL, test_zip)
    
    # Download test ground truth
    test_gt_zip = RAW_DIR / "GTSRB_Final_Test_GT.zip"
    print(f"Downloading test ground truth from {GTSRB_TEST_GT_URL}")
    download_url(GTSRB_TEST_GT_URL, test_gt_zip)
    
    # Extract files
    print("Extracting datasets...")
    
    # Extract training data
    train_extract_dir = RAW_DIR / "train"
    os.makedirs(train_extract_dir, exist_ok=True)
    extract_zip(train_zip, train_extract_dir)
    
    # Extract test data
    test_extract_dir = RAW_DIR / "test"
    os.makedirs(test_extract_dir, exist_ok=True)
    extract_zip(test_zip, test_extract_dir)
    
    # Extract test ground truth
    extract_zip(test_gt_zip, test_extract_dir)
    
    print("Download and extraction completed successfully!")
    print(f"Raw data is available at: {RAW_DIR}")
    print("Next steps:")
    print("1. Run the preprocessing script to prepare the data for training")
    print("2. Train the model using the training script")

if __name__ == "__main__":
    main() 