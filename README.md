# Ambulance Detection using YOLOv8

This repository contains a YOLOv8-based object detection model trained to detect ambulances in images and videos. It includes trained weights, sample input/output data, and the dataset used for training.

## Table of Contents
1. [Model Overview](#model-overview)
2. [Dataset](#dataset)
3. [Trained Weights](#trained-weights)
4. [Sample Input/Output](#sample-inputoutput)
5. [Notebook](#notebook)
6. [Usage](#usage)

---

## Model Overview
The model is trained using the YOLOv8 architecture to detect ambulances in various scenarios. It achieves high accuracy on both close-up and distant views of ambulances.

---

## Dataset
The dataset used for training and validation is organized as follows:
- **Images**:
  - Training: `Main dataset/AMb2.zip`
- **Labels**:
  - Corresponding label files in YOLO format (bounding box annotations).

---

## Trained Weights
The following pre-trained weights are provided:
- `Models/best.pt`: Best-performing model checkpoint based on validation metrics.
- `Models/last.pt`: Model checkpoint saved at the end of the last epoch.

---

## Sample Input/Output
### Images
- **Input Images**: `Sample Images/input imgs/`
- **Output Images**: `Sample Images/output imgs/`

### Videos
- **Input Videos**: `Sample Videos/input vid/`
- **Output Videos**: `Sample Videos/output vid/`

---

## Notebook
The project workflow is documented in the following Jupyter Notebook:
- `notebooks/Ambulance_detection_v8.ipynb`: A comprehensive notebook that includes:
  - Dataset preparation.
  - Model training and evaluation.
  - Inference on images and videos.

You can open this notebook in Google Colab or Jupyter Notebook to replicate the results.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Karthik-coder-003/Ambulance-Detection-YOLOv8/blob/main/notebooks/Ambulance_detection_v8.ipynb)

---

## Usage

### Inference on Images
To perform inference on an image, use the following code:

```python
from ultralytics import YOLO

# Load the model
model = YOLO("Models/best.pt")

# Perform inference on an image
results = model.predict(source="path/to/image.jpg", save=True)
```
### Inference on Videos 
To perform inference on a video, use the following code:

```python
from ultralytics import YOLO

# Load the model
model = YOLO("Models/best.pt")

# Perform inference on a video
results = model.predict(source="path/to/video.mp4", save=True)
