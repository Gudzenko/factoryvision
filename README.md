# FactoryVision

Computer vision application for face and person detection using multiple detection methods with real-time AR face effects.

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for camera-based demos)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd factoryvision
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

**Run main detection demo:**
```bash
python main.py
```

**Run AR face effects demo:**
```bash
python face_ar_demo.py
```

Press `ESC` to exit any demo.

## Features

- Multiple camera support
- Real-time detection and visualization
- Pluggable detector architecture
- Multiple detection methods (face, body, pose, hands)
- AR face effects with real-time tracking

## Detection Methods Comparison

| Detector         | Type         | Speed      | Accuracy   | Distance Range | Resource Usage | Advantages                                                                                  | Disadvantages                                                             | Best Use Case                  |
| ---------------- | ------------ | ---------- | ---------- | -------------- | -------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------ |
| **Haar Cascade** | Face         | ⭐⭐⭐⭐⭐ | ⭐⭐       | Close          | Minimal        | ✅ Very fast<br>✅ No downloads<br>✅ Low resource usage                                    | ❌ Poor with rotation<br>❌ Many false positives<br>❌ Lighting sensitive | Simple tasks, speed critical   |
| **DNN Face**     | Face         | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | Any            | Low            | ✅ Works with rotation<br>✅ Stable lighting<br>✅ Good accuracy<br>✅ Fast enough          | ⚠️ Model download (~2MB)                                                  | General face detection         |
| **YOLO v11**     | Person/Multi | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | Any            | Medium-High    | ✅ Excellent accuracy<br>✅ Full-body detection<br>✅ 80+ object classes<br>✅ Any distance | ❌ More resources<br>⚠️ Model download (~6MB)                             | Person detection, multi-object |
| **MediaPipe**    | Face         | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | Close-Medium   | Minimal        | ✅ Real-time optimized<br>✅ Very fast<br>✅ Excellent close range<br>✅ Lightweight        | ❌ Weaker at distance<br>⚠️ Two models (0-2m / 0-5m)                      | Webcam, selfie, close-range    |

## Recommendations by Scenario

### Face Detection

| Scenario               | Recommended Detector     | Reason                                   |
| ---------------------- | ------------------------ | ---------------------------------------- |
| **Webcam / Selfie**    | MediaPipe Face Detection | Optimized for close range, very fast     |
| **Any Distance**       | DNN Face Detector        | Reliable at all distances, good accuracy |
| **Minimal Resources**  | Haar Cascade             | Fastest, but lowest accuracy             |
| **Production Quality** | DNN Face Detector        | Best balance of speed and accuracy       |

### Person/Body Detection

| Scenario                   | Recommended Detector | Reason                                    |
| -------------------------- | -------------------- | ----------------------------------------- |
| **Full Body Detection**    | YOLO v11             | Most accurate for complete person         |
| **Multi-Object Scenarios** | YOLO v11             | Detects 80+ object classes simultaneously |
| **Real-Time Performance**  | YOLO v11 nano        | Optimized small model                     |

## Switching Between Detectors

All detectors implement a common interface and can be easily switched using `DetectorFactory`. Simply specify the detector type and pass any additional parameters:

```python
from face_body_detectors import DetectorFactory, DetectorType

# Switch between detectors
detector = DetectorFactory.create(DetectorType.MEDIAPIPE, logger=logger, model_selection=1)
detector = DetectorFactory.create(DetectorType.DNN_FACE, logger=logger)
detector = DetectorFactory.create(DetectorType.YOLO, logger=logger, target_classes=[0])
detector = DetectorFactory.create(DetectorType.HAAR_CASCADE, logger=logger)
```

Available types: `HAAR_CASCADE`, `DNN_FACE`, `YOLO`, `MEDIAPIPE`

## Switching Between Video Sources

The application supports both live camera feed and video file playback through `SourceFactory`. Switch between sources easily:

```python
from utils import SourceFactory, SourceType

# Live camera (default camera ID 0)
source = SourceFactory.create(SourceType.CAMERA)

# Specific camera ID
source = SourceFactory.create(SourceType.CAMERA, camera_id=1, logger=logger)

# Video file - fast mode (no delay)
source = SourceFactory.create(SourceType.VIDEO_FILE, 
                             video_path="video.mp4", 
                             loop=True, 
                             realtime=False)

# Video file - realtime mode (with FPS delay)
source = SourceFactory.create(SourceType.VIDEO_FILE, 
                             video_path="video.mp4", 
                             loop=True, 
                             realtime=True, 
                             logger=logger)

# Video file - custom speed (0.8 = 20% faster)
source = SourceFactory.create(SourceType.VIDEO_FILE, 
                             video_path="video.mp4", 
                             loop=True, 
                             realtime=True, 
                             speed_factor=0.8, 
                             logger=logger)
```

**Parameters:**
- `loop` — restart video from beginning when finished (default: True)
- `realtime` — playback with original FPS timing (default: False)
- `speed_factor` — speed multiplier: 1.0=normal, 0.5=2x faster, 2.0=2x slower (default: 1.0)

Available types: `CAMERA`, `VIDEO_FILE`

## Keypoint Detection Methods Comparison

| Detector                  | Type       | Points Count | Speed      | Accuracy   | Resource Usage | Advantages                                                                                      | Disadvantages                                                      | Best Use Case                       |
| ------------------------- | ---------- | ------------ | ---------- | ---------- | -------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------- |
| **MediaPipe Pose**        | Body Pose  | 33           | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | Minimal        | ✅ Real-time optimized<br>✅ Face + body landmarks<br>✅ Very fast<br>✅ Lightweight            | ❌ Single person only<br>❌ Weaker at distance                     | Fitness, yoga, gesture control      |
| **MediaPipe Hands**       | Hand       | 21 per hand  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Minimal        | ✅ Excellent hand tracking<br>✅ Finger details<br>✅ Left/Right detection<br>✅ Up to 2 hands  | ❌ Requires visible hands<br>❌ Struggles with occlusion           | Sign language, hand gestures        |
| **MediaPipe Face Mesh**   | Face       | 468          | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Low            | ✅ Detailed face map<br>✅ Eyes, lips, contours<br>✅ 3D landmarks<br>✅ Refine mode available  | ⚠️ High point count<br>❌ Close-range focused                      | AR filters, face animation          |
| **YOLO Pose**             | Body Pose  | 17           | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | Medium-High    | ✅ Multiple people<br>✅ Works at any distance<br>✅ Excellent accuracy<br>✅ Robust to occlusion | ❌ More resources<br>❌ No face details<br>⚠️ Model download (~6MB) | Crowd analysis, sports, surveillance|

## Recommendations by Keypoint Scenario

### Body Pose Detection

| Scenario                     | Recommended Detector | Reason                                        |
| ---------------------------- | -------------------- | --------------------------------------------- |
| **Single Person (Close)**    | MediaPipe Pose       | Fastest, includes face landmarks              |
| **Multiple People**          | YOLO Pose            | Only option for multi-person detection        |
| **Fitness/Yoga Apps**        | MediaPipe Pose       | Real-time, low latency, detailed points       |
| **Sports Analytics**         | YOLO Pose            | Works at distance, multiple athletes          |
| **Any Distance**             | YOLO Pose            | Consistent accuracy regardless of distance    |

### Hand Detection

| Scenario                     | Recommended Detector | Reason                                        |
| ---------------------------- | -------------------- | --------------------------------------------- |
| **Hand Gestures**            | MediaPipe Hands      | Only hand detector available, excellent       |
| **Sign Language**            | MediaPipe Hands      | Detailed finger tracking, left/right labels   |
| **AR Hand Filters**          | MediaPipe Hands      | Real-time performance, precise landmarks      |

### Face Landmark Detection

| Scenario                     | Recommended Detector    | Reason                                     |
| ---------------------------- | ----------------------- | ------------------------------------------ |
| **Face Animation**           | MediaPipe Face Mesh     | 468 points, detailed mapping               |
| **AR Face Filters**          | MediaPipe Face Mesh     | Eyes, lips, contours tracked precisely     |
| **Emotion Detection**        | MediaPipe Face Mesh     | Detailed facial features                   |
| **Basic Face Pose**          | MediaPipe Pose          | If body pose needed too (includes 5 face)  |

## Switching Between Keypoint Detectors

All keypoint detectors implement a common interface through `KeypointDetectorFactory`:

```python
from pose_hand_detectors import KeypointDetectorFactory, KeypointDetectorType

# Body pose detection (33 landmarks)
detector = KeypointDetectorFactory.create(
    KeypointDetectorType.MEDIAPIPE_POSE, 
    logger=logger, 
    model_complexity=1
)

# Hand detection (up to 2 hands, 21 points each)
detector = KeypointDetectorFactory.create(
    KeypointDetectorType.MEDIAPIPE_HANDS, 
    logger=logger, 
    max_num_hands=2
)

# Face mesh (468 landmarks)
detector = KeypointDetectorFactory.create(
    KeypointDetectorType.MEDIAPIPE_FACE_MESH, 
    logger=logger, 
    max_num_faces=1,
    refine_landmarks=True
)

# YOLO Pose (17 COCO keypoints, multiple people)
detector = KeypointDetectorFactory.create(
    KeypointDetectorType.YOLO_POSE, 
    logger=logger, 
    model_name='yolo11n-pose.pt'
)
```

Available types: `MEDIAPIPE_POSE`, `MEDIAPIPE_HANDS`, `MEDIAPIPE_FACE_MESH`, `YOLO_POSE`

## AR Face Effects

The application includes real-time AR effects that can be applied to detected faces using MediaPipe Face Mesh (468 landmarks). Effects automatically track face position, rotation, and scale.

### Available Effects

| Effect               | Description                                    | Key Parameters                                          | Use Case                           |
| -------------------- | ---------------------------------------------- | ------------------------------------------------------- | ---------------------------------- |
| **GlassesEffect**    | Sunglasses or eyewear overlay                  | `scale_factor` (default: 1.8)                           | Fun filters, accessories           |
| **HatEffect**        | Hat or headwear positioned above forehead      | `scale_factor` (2.2), `x_offset` (0.0), `y_offset` (-0.5) | Holiday themes, costume effects    |
| **FullFaceMaskEffect** | Full face mask (beard, hat, entire face overlay) | `scale_factor` (2.5), `x_offset` (0.0), `y_offset` (0.0) | Character transformations, themes  |
| **FrameEffect**      | Decorative border around entire frame          | None (auto-scales to frame size)                        | Photo booth, branding              |

### Effect Features

All face-tracking effects include:
- ✅ **Automatic rotation** — follows head tilt and turns
- ✅ **Precise positioning** — tracks facial landmarks in real-time
- ✅ **No clipping** — rotated images expand canvas to prevent cutoff
- ✅ **Alpha blending** — smooth transparency overlay
- ✅ **Adjustable offsets** — fine-tune position with x/y parameters

### Using AR Effects

Effects can be combined and applied in sequence:

```python
from ar_effects import GlassesEffect, HatEffect, FullFaceMaskEffect, FrameEffect
from pose_hand_detectors import KeypointDetectorFactory, KeypointDetectorType

# Initialize face detector
face_detector = KeypointDetectorFactory.create(
    KeypointDetectorType.MEDIAPIPE_FACE_MESH,
    max_num_faces=1,
    refine_landmarks=True
)

# Create effect pipeline
effects = [
    FrameEffect("assets/images/frame.png"),
    GlassesEffect("assets/images/glasses.png", scale_factor=1.8),
    HatEffect("assets/images/hat.png", scale_factor=2.2, x_offset=0.0, y_offset=-0.5),
]

# Apply effects
face_detections = face_detector.detect(frame)
for effect in effects:
    frame = effect.apply(frame, face_detections)
```

### Parameter Guide

**Scale Factor:**
- Controls size relative to face/frame
- `< 1.0` — smaller
- `1.0` — same size as reference measurement
- `> 1.0` — larger (typical: 1.5-2.5)

**X Offset:**
- Horizontal shift along face orientation
- `0.0` — centered (default)
- `> 0` — shift right (relative to face)
- `< 0` — shift left

**Y Offset:**
- Vertical shift perpendicular to face orientation
- `0.0` — centered on reference point
- `> 0` — shift down
- `< 0` — shift up (typical for hats: -0.3 to -0.7)

### Creating Custom Effects

Extend `BaseAREffect` to create your own:

```python
from ar_effects import BaseAREffect
import cv2
import numpy as np

class CustomEffect(BaseAREffect):
    def __init__(self, image_path, **kwargs):
        self.img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    def apply(self, frame, detections):
        # Your effect logic here
        return frame
```

All effects receive face detections with 468 MediaPipe Face Mesh landmarks. Key landmark indices:
- Eyes: `33, 133, 362, 263`
- Forehead: `10, 151`
- Temples: `234, 454`
- Chin: `152`

## License

See LICENSE file for details.
