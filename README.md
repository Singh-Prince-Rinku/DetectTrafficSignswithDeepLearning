# 🚦 Deep Learning-Based Traffic Sign Detection

<div align="center">

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-green)
![GTSRB](https://img.shields.io/badge/dataset-GTSRB-red)

*A powerful, real-time traffic sign detection and classification system using state-of-the-art YOLOv8 architecture.*

[Features](#-features) • 
[Quick Start](#-quick-start) • 
[Installation](#%EF%B8%8F-installation) • 
[Dataset](#-dataset) • 
[Training](#-training) • 
[Evaluation](#-evaluation) • 
[Results](#-results) • 
[License](#-license)

<img src="https://i.imgur.com/pFIOryH.png" alt="Traffic Sign Detection Demo" width="600"/>

</div>

## 🔍 Features

- **State-of-the-art Detection**: Built on YOLOv8, offering excellent balance between speed and accuracy
- **Multi-class Recognition**: Identifies and classifies 43 different traffic sign categories
- **Real-time Processing**: Optimized for processing video streams with high FPS
- **Comprehensive Metrics**: Detailed evaluation with mAP, IoU, Precision, Recall metrics
- **Data Augmentation**: Robust training with rotation, scaling, brightness adjustments
- **Easy Deployment**: Simple-to-use scripts for inference on images, videos, or webcam feeds

## 🚀 Quick Start

Run inference on an image:
```bash
python detect.py --source examples/00000_00000.jpg --weights models/best_yolov8s_gtsrb.pt
```

Or use your webcam:
```bash
python detect.py --source 0 --weights models/best_yolov8s_gtsrb.pt
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd traffic-sign-detection
   ```

2. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset**:
   ```bash
   python utils/download_dataset.py
   python utils/preprocess_dataset.py
   python convert_images.py  # Convert images to compatible format
   ```

4. **Download pre-trained model weights**:
   ```bash
   python download_model.py
   ```

## 📂 Project Structure

```
.
├── data/                     # Dataset storage
│   ├── raw/                  # Raw dataset files
│   └── processed/            # Processed dataset (YOLO format)
├── models/                   # Model weights
├── utils/                    # Utility functions
├── configs/                  # Configuration files
├── examples/                 # Example images
├── results/                  # Inference results
├── train.py                  # Training script
├── train_yolo.py             # Simplified training script
├── evaluate.py               # Evaluation script
├── detect.py                 # Inference script for images/videos
└── requirements.txt          # Project dependencies
```

## 📊 Dataset

This project utilizes the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset:

<div align="center">
<table>
  <tr>
    <td align="center"><b>50,000+</b><br>images</td>
    <td align="center"><b>43</b><br>classes</td>
    <td align="center"><b>Various</b><br>lighting conditions</td>
    <td align="center"><b>Different</b><br>viewing angles</td>
  </tr>
</table>
</div>

The dataset undergoes comprehensive preprocessing:
- ✅ Conversion to YOLO format
- ✅ Resizing to uniform dimensions
- ✅ Normalization
- ✅ Data augmentation

## 🧠 Training

Train the model with default settings:
```bash
python train_yolo.py
```

Or customize your training:
```bash
python train_yolo.py --epochs 100 --batch-size 16 --img-size 640 --model-size s
```

Parameters are configured in `configs/train_config.yaml`:
```yaml
model:
  type: 'yolov8'
  size: 's'  # Model size (n, s, m, l, x)
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  # ... and more
```

## 📈 Evaluation

Evaluate your trained model:
```bash
python evaluate.py --weights models/best_yolov8s_gtsrb.pt
```

This produces:
- Precision, recall, and F1 score metrics
- Mean Average Precision (mAP) calculations
- Per-class performance analysis
- Confusion matrix visualization

## ✨ Results

<div align="center">
<table>
  <tr>
    <th>Model</th>
    <th>mAP@0.5</th>
    <th>mAP@0.5:0.95</th>
    <th>Inference Speed</th>
  </tr>
  <tr>
    <td>YOLOv8s</td>
    <td>0.379</td>
    <td>0.324</td>
    <td>~470ms/image (CPU)</td>
  </tr>
</table>
</div>

<div align="center">
<img src="https://i.imgur.com/Y4yZkVq.png" alt="Example Results" width="400"/>
</div>

## 🔧 Usage Examples

### Image Detection
```bash
python detect.py --source examples/00000_00000.jpg --weights models/best_yolov8s_gtsrb.pt
```

### Video Processing
```bash
python detect.py --source path/to/video.mp4 --weights models/best_yolov8s_gtsrb.pt
```

### Webcam Mode
```bash
python detect.py --source 0 --weights models/best_yolov8s_gtsrb.pt
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Citation

If you use the GTSRB dataset in your work, please cite:
```
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. 
The German Traffic Sign Recognition Benchmark: A multi-class classification competition.
In Proceedings of the IEEE International Joint Conference on Neural Networks, 
pages 1453–1460. 2011.
```

## 🌟 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [GTSRB](https://benchmark.ini.rub.de/gtsrb_news.html) for providing the dataset
- [OpenCV](https://opencv.org/) for image processing capabilities 