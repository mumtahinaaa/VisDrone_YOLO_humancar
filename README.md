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

### Install dependencies
```bash
pip install ultralytics opencv-python
```

### Detection & Counting
```python
from ultralytics import YOLO
import cv2

model = YOLO('weights/best.pt')

results = model.predict(
    source='your_image.jpg',
    classes=[0, 1, 3],  # pedestrian, people, car
    conf=0.25
)

for r in results:
    class_ids = r.boxes.cls.tolist()
    human_count = class_ids.count(0) + class_ids.count(1)
    car_count = class_ids.count(3)
    print(f"Humans: {human_count}, Cars: {car_count}")
    r.save(filename='output.jpg')
```

### Training
```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data='visdrone.yaml',
    epochs=50,
    imgsz=960,
    batch=8,
    patience=10,
    device=0
)
```
## Project Structure
VisDrone_YOLO_humancar/
├── README.md
├── outputs/          # Sample detection results
├── weights/          # Trained model weights (best.pt)
└── notebooks/        # Colab training notebook
## Dataset
- Source: VisDrone 2019 Detection Dataset
- Train: 6,471 images
- Val: 548 images
- Classes: 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- Target classes at inference: pedestrian (0), people (1), car (3)

## Challenges
- Extremely small objects due to high altitude drone footage
- Heavy occlusion in dense crowd scenes
- Class imbalance (car: 14,064 instances vs bus: 251 instances)
- Ambiguous pedestrian vs people separation

## Limitations
- Model struggles with heavily occluded tiny humans
- Tracking not completed due to GPU session limits
- Higher resolution (1280px) could improve small object detection further
python detect.py
```
