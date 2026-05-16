# VisDrone_YOLO_humancar
## VisDrone Human & Car Detection System
Antlings Internship Assessment — AI/ML Drone Human Detection & Counting

## Overview
A computer vision pipeline for detecting humans and cars in drone/aerial imagery using YOLOv8, trained on the VisDrone 2019 dataset. The system detects pedestrians, people, and cars — counting total humans and visualizing results with bounding boxes.

## Tasks Completed
- Task 01: Dataset Understanding & Preprocessing
- Task 02: Model Training (YOLOv8s)
- Task 03: Human & Car Detection with Counting
- Task 04: Object Tracking (ByteTrack) (Bonus Task)
- Task 05: Evaluation & Visualization

## Real Workflow & Engineering Decisions

### Environment Setup
Getting the environment right took some trial and error. We started with Kaggle but hit a wall — phone verification was required to enable internet, which blocked package installation. Switched to Google Colab, but the free tier disconnected mid-training and we lost the weights entirely. Eventually settled on Kaggle with T4 x2 GPU after getting verification sorted, which gave us stable training with 30 hours of free GPU time. Key lesson: always download best.pt the moment training finishes.

### (Task 01) Dataset Understanding & Preprocessing
VisDrone 2019 is a large-scale aerial detection dataset with 6,471 training images, 548 validation images, and 10 object categories. The dataset comes fully pre-labeled, so no manual annotation was needed. Each image has a corresponding label file containing bounding box coordinates and class IDs.

The dataset presented several real challenges. Objects are extremely small because the drone is at high altitude — people are sometimes just a few pixels wide. Dense crowds and parking lots cause heavy occlusion. There's also a significant class imbalance: cars appear 14,064 times in the validation set while buses appear only 251 times. An interesting quirk of VisDrone is that humans are split into two separate classes — "pedestrian" (walking) and "people" (standing, sitting, in groups) — which complicates counting.

For preprocessing, we relied entirely on YOLOv8's built-in augmentation pipeline: mosaic (stitches 4 images together), horizontal flipping, random scaling, and HSV jitter for lighting variation. This handled everything automatically during training.

### (Task 02) Model Training
We chose YOLOv8s as the base model — it strikes the best balance between speed and accuracy for real-time drone applications, and comes pretrained on COCO which gave us a strong starting point.

The first training run used the default imgsz=640, but we stopped it early after realizing that aggressively downsampling high-resolution aerial footage causes the model to miss tiny objects. Restarted with imgsz=960, which gave the model much more detail to work with. We also made a deliberate decision to train on all 10 classes rather than filtering the labels beforehand — this preserved dataset integrity and eliminated any risk of corrupting the annotation files. Final config: imgsz=960, batch=8, epochs=50, patience=10 for early stopping.

### (Task 03) Human & Car Detection with Counting
Rather than retraining a separate model for just humans and cars, we used YOLO's native classes argument at inference time — passing classes=[0, 1, 3] to filter detections to pedestrian, people, and car only. This is cleaner, faster, and carries no risk of data manipulation.

Since VisDrone splits humans into two classes, the total human count combines both: pedestrian detections + people detections. Bounding boxes and the human count overlay are drawn directly onto the output images using OpenCV.

Sample results across 5 test images:
Image 1: Humans=1, Cars=41
Image 2: Humans=5, Cars=14
Image 3: Humans=40, Cars=0
Image 4: Humans=1, Cars=5
Image 5: Humans=4, Cars=42

### (Task 04) Object Tracking
ByteTrack was implemented using Ultralytics' built-in tracker — literally one line of code change from predict to track. We first tested it on VisDrone image sequences, but the result looked like a slideshow because the test-dev folder mixes images from completely different drone flights. We attempted to filter by sequence ID "0000001_" but found no matches in the test-dev set, so fell back to a sorted sequence. To make the output more visually clear, we used frame duplication to slow playback down.

For a more convincing demo, we switched to a real continuous drone video where ByteTrack could actually maintain tracking IDs across frames naturally. We also compared ByteTrack against BoT-SORT — both performed similarly, but ByteTrack was kept as it is the more widely adopted standard. OC-SORT was considered for its stronger handling of fast-moving objects but skipped due to time constraints.

### (Task 05) Evaluation & Visualization

| Class | mAP@0.5 | Precision | Recall |
|---|---|---|---|
| All | 47.6% | 58.0% | 47.6% |
| Pedestrian | 55.2% | 62.8% | 53.4% |
| People | 42.4% | 63.0% | 39.6% |
| Bicycle | 24.6% | 38.4% | 28.4% |
| Car | 84.3% | 78.5% | 82.9% |
| Van | 51.6% | 57.3% | 52.2% |
| Truck | 44.6% | 57.8% | 43.7% |
| Bus | 61.2% | 76.3% | 53.9% |
| Motor | 56.5% | 61.3% | 57.0% |

Car detection came out exceptionally strong at 84.3% mAP, which makes sense given cars are the most represented class. Human detection (pedestrian + people combined) sits around 50%, which is reasonable for such a challenging aerial dataset. Inference runs at approximately 169 FPS, making it well-suited for real-time applications. All metrics were extracted directly from the model output rather than hardcoded. Confusion matrix, PR curve, F1 curve saved from training run

## Strengths
- Excellent car detection (84.3% mAP)
- Fast inference suitable for real-time use (~169 FPS)
- Clean inference filtering without touching training data
- ByteTrack integration with minimal code

## Limitations
- Struggles with heavily occluded tiny humans
- Tracking IDs inconsistent on image sequences (not real video)
- Higher resolution (1280px) could improve small object detection further
- People vs pedestrian ambiguity affects human counting accuracy

## Tech Stack
- Detection: YOLOv8s
- Tracking: ByteTrack
- Visualization: OpenCV
- Training: PyTorch + Kaggle T4 x2 GPU

## Project Structure
```
VisDrone_YOLO_humancar/
├── README.md
├── outputs/          # Sample detection results and tracking videos
├── weights/          # Trained model weights (best.pt)
└── notebooks/        # Kaggle training notebook
```

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
    human_count = sum(1 for c in class_ids if c in [0, 1])
    print(f"Total Humans: {human_count}")
    r.save(filename='output.jpg')
```

### Tracking
```python
model.track(
    source='drone_video.mp4',
    classes=[0, 1, 3],
    tracker="bytetrack.yaml",
    persist=True
)
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

## Dataset
- Source: [VisDrone 2019 Detection Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
- Train: 6,471 images
- Val: 548 images
- Classes: 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- Target classes at inference: pedestrian (0), people (1), car (3)
- Higher resolution (1280px) could improve small object detection further
python detect.py
```
