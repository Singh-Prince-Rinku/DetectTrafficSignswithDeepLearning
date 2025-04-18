# Deep Learning-Based Traffic Sign Detection System

This project implements a deep learning-based system for detecting and classifying traffic signs in images and video streams in real-time.

## Features

- Traffic sign detection and classification using YOLOv8
- Support for both static images and real-time video processing
- Comprehensive evaluation metrics (mAP, IoU, Precision, Recall)
- Data augmentation techniques for improved model robustness
- Easy-to-use inference scripts for deployment

## Project Structure

```
.
├── data/                     # Dataset storage
│   ├── raw/                  # Raw dataset files
│   └── processed/            # Processed dataset
├── models/                   # Model definitions and weights
├── utils/                    # Utility functions
├── configs/                  # Configuration files
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── detect.py                 # Inference script for images/videos
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd traffic-sign-detection
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download the dataset (GTSRB by default):
   ```
   python utils/download_dataset.py
   ```

## Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset by default. The dataset contains:
- Over 50,000 images of traffic signs
- 43 different classes
- Images of varying sizes and lighting conditions

Data preprocessing steps include:
- Resizing to a uniform size
- Normalization
- Data augmentation (rotation, scaling, brightness adjustment, etc.)

## Training

To train the model:

```
python train.py --config configs/train_config.yaml
```

The training configuration can be customized in the YAML file.

## Evaluation

To evaluate the trained model:

```
python evaluate.py --weights models/best.pt --data data/processed/test
```

## Inference

For detecting traffic signs in images:

```
python detect.py --source path/to/image.jpg --weights models/best.pt
```

For video processing:

```
python detect.py --source path/to/video.mp4 --weights models/best.pt
```

For webcam:

```
python detect.py --source 0 --weights models/best.pt
```

## Model Architecture

This project uses YOLOv8, a state-of-the-art deep learning object detection model that offers an excellent balance between accuracy and inference speed. The model has been fine-tuned specifically for traffic sign detection.

## Performance

The model achieves:
- mAP@0.5: TBD after training
- Inference speed: TBD frames per second on GPU

## License

[Specify the license here]

## Citation

If you use the GTSRB dataset, please cite:
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011. 