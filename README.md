# VisDrone_YOLO_humancar
# VisDrone Human & Car Detection System
Antlings Internship Assessment — AI/ML Drone Human Detection & Counting

## Overview
A computer vision pipeline for detecting humans and cars in drone/aerial imagery using YOLOv8, trained on the VisDrone 2019 dataset.

## Tasks Completed
- ✅ Task 01: Dataset Understanding & Preprocessing
- ✅ Task 02: Model Training (YOLOv8s)
- ✅ Task 03: Human & Car Detection with Counting
- ⭐ Task 04: Object Tracking (ByteTrack)
- ✅ Task 05: Evaluation & Visualization

## Model & Results
| Metric | Value |
|---|---|
| mAP@0.5 (all classes) | 47.6% |
| mAP@0.5 (car) | 84.3% |
| mAP@0.5 (pedestrian) | 55.2% |
| Precision | 58% |
| Recall | 47.6% |
| Inference Speed | ~169 FPS |

## Tech Stack
- Detection: YOLOv8s
- Tracking: ByteTrack
- Visualization: OpenCV
- Training: PyTorch + Google Colab T4 GPU

## Dataset
VisDrone 2019 — 6,471 training images, 548 validation images, 10 classes

## Key Design Decisions
- Trained on all 10 classes, filtered to humans/cars at inference using `classes=[0,1,3]`
- Used imgsz=960 instead of default 640 for better small object detection
- Early stopping with patience=10 to prevent overfitting

## Sample Results
![Detection Result](outputs/result.jpg)

## How to Run
```bash
pip install ultralytics
python detect.py
```
pip install ultralytics
python detect.py
```
