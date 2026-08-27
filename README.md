# Sentinel  Real-Time Violence & Threat Detection for Public Spaces

An edge-deployable spatiotemporal AI model that watches public space video feeds and detects violent activity — fighting, shootings, assault, armed threats — as it's unfolding, and alerts authorities before it escalates.

---

## Overview

Sentinel is built for public safety monitoring — transit stations, malls, streets, campuses — anywhere a fixed or networked camera feed exists but no one can watch every screen at once. Instead of relying on a human operator to catch a fight or a weapon in a sea of monitors, Sentinel watches continuously and flags violent activity the moment its confidence crosses threshold, routing an alert to the relevant authority in near real time.

The core of Sentinel is STEAD, a spatiotemporal deep learning model built on an X3D backbone, trained specifically to recognize the motion and appearance patterns of violent activity across video clips rather than single frames — because violence is an event that unfolds over time, not a static image.

Built as a personal project to explore real-time video understanding for public safety, using models efficient enough to run continuously on modest hardware rather than requiring a data-center GPU per camera.

---

## Demo

[▶️ Watch Titan in Action](./media/down.mp4)


---

## Key Features

- Spatiotemporal Violence Detection: Analyzes short video clips (not single frames) to catch actions that only make sense over time — a punch thrown, a struggle, a weapon drawn.
- X3D Backbone: Uses X3D, an efficient 3D-CNN architecture, as the feature extractor, chosen for its strong accuracy-to-compute ratio on video classification tasks.
- STEAD Classification Head: A custom-trained spatiotemporal head (STEAD) sits on top of the X3D features to classify clips into violent vs. non-violent, and further into specific categories (fighting, shooting, assault, robbery).
- Trained on Real-World Crime Footage: Fine-tuned on the UCF-Crime dataset — real surveillance footage spanning categories like abuse, arrest, arson, assault, burglary, explosion, fighting, robbery, shooting, and vandalism.
- Continuous Public-Space Monitoring: Designed to run on a persistent video stream, not just isolated clips, scanning a rolling window for events as they happen.
- Authority Alerting: Confirmed violent-event detections trigger an alert pipeline to notify designated authorities/monitoring staff with a timestamped clip and location.

---

## How It Works

### 1. Video Ingestion

Sentinel ingests a live video stream from a public-space camera and continuously buffers it into overlapping short clips (e.g. a few seconds each), the unit of analysis the model actually reasons over — violence is a temporal pattern, so single frames aren't enough.

### 2. Feature Extraction — X3D

Each clip is passed through X3D, a 3D-convolutional video backbone that extracts spatiotemporal features — capturing both what's in the frame and how it's changing across frames (motion, trajectory, interaction between people). X3D was chosen over heavier 3D-CNN alternatives for its efficiency, making sustained real-time inference on modest hardware feasible.

### 3. Classification — STEAD

The X3D features feed into STEAD, the trained classification head, which outputs:
- A binary violent / non-violent confidence score
- A category label when violence is detected (fighting, shooting, assault, robbery, etc.), based on the UCF-Crime category taxonomy the model was fine-tuned on

STEAD was trained on the UCF-Crime dataset, a real-world surveillance-footage dataset built specifically for anomaly and crime detection research, giving the model exposure to the visual and motion signatures of real violent incidents rather than staged or synthetic footage.

### 4. Alert Pipeline

When a clip's violence confidence crosses the configured threshold:

1. The triggering clip and its classification (category, confidence score, timestamp, camera/location ID) are packaged into an alert.
2. The alert is sent to the configured authority-facing endpoint (e.g. a monitoring dashboard or notification service).
3. A short buffer of preceding and following frames is retained alongside the alert, giving the responding authority immediate context without needing to pull raw footage separately.

---

## Architecture

Video Source (public-space camera / stream)
- Continuous feed, buffered into overlapping short clips

Inference Pipeline
- X3D — spatiotemporal feature extraction per clip
- STEAD — classification head (violent / non-violent + category)
- Threshold & Confirmation Logic — reduces false positives from crowding, sports, horseplay, etc.

Alert Manager
- Packages triggering clip + metadata (category, confidence, timestamp, location)
- Sends to authority-facing endpoint / monitoring dashboard

---

## Dataset

- UCF-Crime Dataset — a large-scale, real-world surveillance video dataset for anomaly/crime detection, covering 13 categories of real criminal and violent activity (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Robbery, Shooting, Shoplifting, Stealing, Vandalism, plus normal/non-violent footage).
- Source: https://huggingface.co/datasets/jinjiyese/UCF-Crime (replace with the exact Hugging Face dataset link you used)

---

## Software Stack

- Backbone: X3D (video feature extraction)
- Classification head: STEAD (custom-trained spatiotemporal classifier)
- Training framework: PyTorch
- Dataset source: Hugging Face Datasets (UCF-Crime)
- Video I/O: OpenCV
- Language(s): Python 3

---

## Getting Started

### Prerequisites

- Python 3.9+, pip
- PyTorch (with CUDA if running on GPU)
- Access to a video stream or recorded footage for inference

### Installation

```
git clone https://github.com/<emmanueloeng>/sentinel.git
cd sentinel

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```
cp config.example.yaml config.yaml
# Edit config.yaml to set:
# - video source (RTSP stream / camera index / file path)
# - violence confidence threshold
# - alert endpoint
# - clip buffer length and overlap
```

### Running Sentinel

```
python main.py
```

### Training / Fine-Tuning

```
python train.py --dataset ucf_crime --backbone x3d --epochs [N]
```

---

## Performance

Fill in with your real numbers once benchmarked — these carry real weight with employers.

- Inference latency per clip: [Xms]
- Violence detection accuracy on UCF-Crime test split: [X%]
- False positive rate: [X per hour of normal public-space footage]
- End-to-end alert latency (event to notification): [under X seconds]

---

## Roadmap

- Multi-camera correlation to reduce duplicate alerts for the same incident
- Weapon-specific sub-classification
- On-device quantized deployment for edge cameras
- Historical incident dashboard for authorities
- Category-specific alert routing (e.g. shooting vs. shoplifting go to different response teams)
