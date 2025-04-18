#!/usr/bin/env python3
"""
Download and extract the GTSRB (German Traffic Sign Recognition Benchmark) dataset.
"""

import os
import sys
import zipfile
import tarfile
import shutil
import argparse
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
    """Download a file from url to output_path."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_dir):
    """Extract a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f'Extracting {zip_path}'):
            try:
                zip_ref.extract(member, extract_dir)
            except zipfile.error as e:
                print(f"Error extracting {member.filename}: {e}")

def extract_tar(tar_path, extract_dir):
    """Extract a tar file to the specified directory."""
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        for member in tqdm(tar_ref.getmembers(), desc=f'Extracting {tar_path}'):
            try:
                tar_ref.extract(member, extract_dir)
            except tarfile.ExtractError as e:
                print(f"Error extracting {member.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Download and extract GTSRB dataset')
    parser.add_argument('--data-dir', type=str, default='data', 
                        help='Path to data directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading files if they already exist')
    args = parser.parse_args()
    
    # Update paths based on args
    global DATA_DIR, RAW_DIR, PROCESSED_DIR
    DATA_DIR = Path(args.data_dir)
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Create directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Download training data
    train_zip = RAW_DIR / "GTSRB_Final_Training_Images.zip"
    if not train_zip.exists() or not args.skip_download:
        print(f"Downloading training data from {GTSRB_TRAIN_URL}")
        download_url(GTSRB_TRAIN_URL, train_zip)
    
    # Download test data
    test_zip = RAW_DIR / "GTSRB_Final_Test_Images.zip"
    if not test_zip.exists() or not args.skip_download:
        print(f"Downloading test data from {GTSRB_TEST_URL}")
        download_url(GTSRB_TEST_URL, test_zip)
    
    # Download test ground truth
    test_gt_zip = RAW_DIR / "GTSRB_Final_Test_GT.zip"
    if not test_gt_zip.exists() or not args.skip_download:
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