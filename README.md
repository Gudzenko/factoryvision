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
python examples/face_ar_demo.py
```

**Run background segmentation demo:**

```bash
python examples/segmentation_demo.py
```

**Run style effects demo:**

```bash
python examples/style_effects_demo.py
```

**Run face contour demo:**

```bash
python examples/face_contour_demo.py
```

**Run caricature generator demo:**

```bash
python examples/caricature_demo.py
```

**Run motion detection demo:**

```bash
python examples/motion_detection_app.py
```

Press `ESC` to exit any demo.

## Features

- Multiple camera support
- Real-time detection and visualization
- Pluggable detector architecture
- Multiple detection methods (face, body, pose, hands)
- AR face effects with real-time tracking
- Background segmentation and replacement
- Style transformation effects (cartoon, sketch, edge detection)

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

| Detector                | Type      | Points Count | Speed      | Accuracy   | Resource Usage | Advantages                                                                                        | Disadvantages                                                       | Best Use Case                        |
| ----------------------- | --------- | ------------ | ---------- | ---------- | -------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------ |
| **MediaPipe Pose**      | Body Pose | 33           | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | Minimal        | ✅ Real-time optimized<br>✅ Face + body landmarks<br>✅ Very fast<br>✅ Lightweight              | ❌ Single person only<br>❌ Weaker at distance                      | Fitness, yoga, gesture control       |
| **MediaPipe Hands**     | Hand      | 21 per hand  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Minimal        | ✅ Excellent hand tracking<br>✅ Finger details<br>✅ Left/Right detection<br>✅ Up to 2 hands    | ❌ Requires visible hands<br>❌ Struggles with occlusion            | Sign language, hand gestures         |
| **MediaPipe Face Mesh** | Face      | 468          | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Low            | ✅ Detailed face map<br>✅ Eyes, lips, contours<br>✅ 3D landmarks<br>✅ Refine mode available    | ⚠️ High point count<br>❌ Close-range focused                       | AR filters, face animation           |
| **YOLO Pose**           | Body Pose | 17           | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | Medium-High    | ✅ Multiple people<br>✅ Works at any distance<br>✅ Excellent accuracy<br>✅ Robust to occlusion | ❌ More resources<br>❌ No face details<br>⚠️ Model download (~6MB) | Crowd analysis, sports, surveillance |

## Recommendations by Keypoint Scenario

### Body Pose Detection

| Scenario                  | Recommended Detector | Reason                                     |
| ------------------------- | -------------------- | ------------------------------------------ |
| **Single Person (Close)** | MediaPipe Pose       | Fastest, includes face landmarks           |
| **Multiple People**       | YOLO Pose            | Only option for multi-person detection     |
| **Fitness/Yoga Apps**     | MediaPipe Pose       | Real-time, low latency, detailed points    |
| **Sports Analytics**      | YOLO Pose            | Works at distance, multiple athletes       |
| **Any Distance**          | YOLO Pose            | Consistent accuracy regardless of distance |

### Hand Detection

| Scenario            | Recommended Detector | Reason                                      |
| ------------------- | -------------------- | ------------------------------------------- |
| **Hand Gestures**   | MediaPipe Hands      | Only hand detector available, excellent     |
| **Sign Language**   | MediaPipe Hands      | Detailed finger tracking, left/right labels |
| **AR Hand Filters** | MediaPipe Hands      | Real-time performance, precise landmarks    |

### Face Landmark Detection

| Scenario              | Recommended Detector | Reason                                    |
| --------------------- | -------------------- | ----------------------------------------- |
| **Face Animation**    | MediaPipe Face Mesh  | 468 points, detailed mapping              |
| **AR Face Filters**   | MediaPipe Face Mesh  | Eyes, lips, contours tracked precisely    |
| **Emotion Detection** | MediaPipe Face Mesh  | Detailed facial features                  |
| **Basic Face Pose**   | MediaPipe Pose       | If body pose needed too (includes 5 face) |

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

| Effect                 | Description                                      | Key Parameters                                            | Use Case                          |
| ---------------------- | ------------------------------------------------ | --------------------------------------------------------- | --------------------------------- |
| **GlassesEffect**      | Sunglasses or eyewear overlay                    | `scale_factor` (default: 1.8)                             | Fun filters, accessories          |
| **HatEffect**          | Hat or headwear positioned above forehead        | `scale_factor` (2.2), `x_offset` (0.0), `y_offset` (-0.5) | Holiday themes, costume effects   |
| **FullFaceMaskEffect** | Full face mask (beard, hat, entire face overlay) | `scale_factor` (2.5), `x_offset` (0.0), `y_offset` (0.0)  | Character transformations, themes |
| **FrameEffect**        | Decorative border around entire frame            | None (auto-scales to frame size)                          | Photo booth, branding             |

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

## Background Effects

Real-time background replacement and segmentation using MediaPipe Selfie Segmentation. Automatically separates person from background for virtual background effects similar to video conferencing apps.

### Person Segmentation

**PersonSegmentation** class provides accurate person detection and masking:

```python
from background_effects import PersonSegmentation

# Initialize segmentation (model_selection: 0=general, 1=landscape/webcam)
segmentation = PersonSegmentation(model_selection=1, logger=logger)

# Get soft mask (0.0-1.0 float values)
mask = segmentation.get_mask(frame)

# Get binary mask (0 or 255)
binary_mask = segmentation.get_binary_mask(frame, threshold=0.5)

# Get contours (list of contour points)
contours = segmentation.get_contours(frame, threshold=0.5)

# Visualize mask overlay
result = segmentation.visualize_mask(frame, mask_color=(0, 255, 0), alpha=0.5)

# Visualize contours
result = segmentation.visualize_contours(frame, contour_color=(0, 255, 0), thickness=2)
```

**Key Features:**

- ✅ Real-time performance (30-60 FPS on CPU)
- ✅ Accurate edge detection (hair, clothing details)
- ✅ Two model modes (general/landscape)
- ✅ Soft masks for smooth blending

### Background Replacement

**BackgroundReplacementEffect** replaces background with custom image:

```python
from background_effects import PersonSegmentation, BackgroundReplacementEffect

# Initialize
segmentation = PersonSegmentation(model_selection=1)
bg_effect = BackgroundReplacementEffect("assets/images/background.jpg")

# Apply replacement
mask = segmentation.get_mask(frame)
result = bg_effect.apply(frame, mask)
```

**How it works:**

1. Segmentation creates mask (person=1.0, background=0.0)
2. Background image is resized to match frame size
3. Blending formula: `result = frame × mask + background × (1 - mask)`
4. Person pixels kept, background pixels replaced

**Use Cases:**

- 📹 Virtual backgrounds for video calls
- 🎬 Green screen effects without green screen
- 🖼️ Custom photo backgrounds
- 🎨 Creative video filters

### Segmentation Demo

Run interactive demo with multiple visualization modes:

```bash
python examples/segmentation_demo.py
```

**Controls:**

- `1` — Grayscale mask (shows segmentation confidence)
- `2` — Colored overlay (green highlight on person)
- `3` — Binary mask (black/white threshold)
- `4` — Contours (edge detection)
- `5` — Background replacement (virtual background)
- `0` — Original frame
- `ESC` — Exit

**Configuration:**
Edit `BACKGROUND_PATH` in `segmentation_demo.py` to change virtual background image.

### Model Selection

| Mode              | Resolution | Speed      | Best For                                  |
| ----------------- | ---------- | ---------- | ----------------------------------------- |
| **0 (General)**   | 256×256    | ⭐⭐⭐     | General purpose, any distance             |
| **1 (Landscape)** | 144×256    | ⭐⭐⭐⭐⭐ | Webcam, selfie, close-range (recommended) |

### Tips for Best Results

**Lighting:**

- Even lighting on person and background
- Avoid strong backlighting (silhouettes)
- Minimize shadows

**Distance:**

- Works best 0.5-2 meters from camera
- Landscape mode optimized for webcam distance

**Threshold tuning:**

- Lower (0.3-0.4): Captures more details, may include background
- Default (0.5): Balanced
- Higher (0.6-0.7): Cleaner edges, may lose details

---

## Style Effects

Transform video frames into artistic styles in real-time. All effects support real-time camera streaming with instant switching between effects.

### Available Effects

#### 1. Canny Edge Effect

**Description:** Classic edge detection creating clean contour lines.

**Parameters:**

- `threshold1` (default: 100) — Lower threshold for edge detection (20-120)
- `threshold2` (default: 150) — Upper threshold for edge detection (60-250, typically 2-3× threshold1)
- `invert` (default: True) — Black lines on white background (False: white lines on black)

**Best for:** Technical drawings, contour art, minimalist style

**Tips:**

- Lower thresholds → more detailed lines
- Higher thresholds → only strong edges
- threshold2 = threshold1 × 2.5-3.0 for good results

---

#### 2. Pencil Sketch Effect

**Description:** Creates pencil drawing appearance with natural sketch lines.

**Parameters:**

- `sigma_s` (default: 75) — Filter size, controls detail level (10-100)
- `sigma_r` (default: 0.07) — Contrast threshold for edges (0.01-0.15)
- `shade_factor` (default: 0.1) — Shading intensity (0.01-0.1)

**Best for:** Artistic sketches, portrait drawings

**Tips:**

- Lower sigma_s → cleaner lines, less blur
- Lower sigma_r → sharper contours
- Lower shade_factor → minimal shading, more line focus

---

#### 3. Cartoon Effect

**Description:** Transforms video into cartoon-like appearance with simplified colors and bold outlines.

**Parameters:**

- `d` (default: 5) — Bilateral filter diameter (5-15)
- `sigma_color` (default: 30) — Color filtering strength (10-150)
- `sigma_space` (default: 30) — Spatial filtering strength (10-150)

**Best for:** Animation style, comic book look

**Tips:**

- Higher d, sigma_color, sigma_space → flatter color areas (more cartoonish)
- Lower values → preserve more details
- sigma_color = sigma_space for balanced results

---

#### 4. Adaptive Threshold Effect

**Description:** Creates high-contrast black-and-white manga/comic style.

**Parameters:**

- `block_size` (default: 11) — Size of neighborhood area (odd number, 5-25)
- `C` (default: 2) — Constant subtracted from mean (0-10)
- `method` (default: MEAN_C) — ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
- `invert` (default: False) — Black lines on white (True: white lines on black)

**Best for:** Manga, comic books, high-contrast art

**Tips:**

- Larger block_size → coarser, simpler result
- Higher C → fewer black lines (lighter overall)
- MEAN_C faster, GAUSSIAN_C smoother

---

#### 5. Oil Painting Effect

**Description:** Simulates oil painting with brush strokes and color blending.

**Parameters:**

- `size` (default: 7) — Brush size/stroke size (1-10)
- `dynRatio` (default: 1) — Dynamic range intensity (1-3)

**Best for:** Artistic painting style, impressionism

**Tips:**

- Larger size → bigger, more visible brush strokes
- Requires opencv-contrib-python for best quality
- Falls back to bilateral filter + quantization if xphoto unavailable

---

### Running Style Effects Demo

```bash
python examples/style_effects_demo.py
```

**Controls:**

- `0` — Show original frame (no effect)
- `1` — Canny Edge Effect
- `2` — Pencil Sketch Effect
- `3` — Cartoon Effect
- `4` — Adaptive Threshold Effect
- `5` — Oil Painting Effect
- `ESC` — Exit

**Performance:**
All effects run in real-time (30+ FPS) on modern hardware. Effects 1-2 (edge detection) are fastest, effect 5 (oil painting) is most computationally intensive.

**Customization:**
Edit effect parameters in `style_effects_demo.py` to fine-tune visual output:

```python
_effects = [
    CannyEdgeEffect(threshold1=100, threshold2=150, invert=True),
    PencilSketchEffect(sigma_s=75, sigma_r=0.07, shade_factor=0.1),
    CartoonEffect(d=5, sigma_color=30, sigma_space=30),
    AdaptiveThresholdEffect(block_size=11, C=2),
    OilPaintingEffect(size=7, dynRatio=1),
]
```

---

## Face Contour Detection Demo

Combines person segmentation, bilateral filtering, and adaptive thresholding to extract clean facial contours.

### Key Techniques

**Bilateral Filter** - Removes texture/noise while preserving edges  
**Person Segmentation** - MediaPipe Selfie Segmentation (landscape mode)  
**Morphological Operations** - Cleans mask (MORPH_CLOSE + MORPH_OPEN)  
**Adaptive Threshold** - Block-based contour detection  
**Mask Composition** - Shows contours only on person, white background elsewhere

### Running

```bash
python examples/face_contour_demo.py
```

### Configuration

Edit parameters in `face_contour_demo.py`:

```python
AdaptiveThresholdEffect(
    block_size=9,      # Neighborhood size (odd number)
    C=3,               # Threshold adjustment (lower = more lines)
    method=cv2.ADAPTIVE_THRESH_MEAN_C
)

cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
```

---

## Caricature Generator

AI-based caricature generation using Stable Diffusion XL + LoRA adapter.

### Installation

```bash
pip install diffusers transformers accelerate safetensors peft pillow torch
```

### Usage

```python
from caricature_generator import CaricatureGenerator
import cv2

generator = CaricatureGenerator(device='cpu', lora_weight=0.9)

image = cv2.imread('assets/images/photo1.jpg')
result = generator.generate(
    image,
    prompt="caricature style, big head, exaggerated features",
    strength=0.75
)

cv2.imwrite('assets/images/result.jpg', result)
```

### Running Demo

```bash
python examples/caricature_demo.py
```

**Note:** First run downloads ~6GB model.

---

## HandGestureRecognizer API

### Class: `HandGestureRecognizer`

A class for hand gesture recognition using the Hugging Face model `prithivMLmods/Hand-Gesture-19`.

#### Initialization

```python
recognizer = HandGestureRecognizer(device='cpu', confidence_threshold=0.5)
```

- `device` (str): Device for inference, either 'cpu' or 'cuda'.
- `confidence_threshold` (float): Minimum probability required to return a gesture label. Default is 0.5.

#### Method: `predict`

```python
gesture, probs = recognizer.predict(image)
```

- `image` (np.ndarray or PIL.Image): Input image (BGR/RGB or PIL.Image).
- Returns:
  - `gesture` (str or None): Predicted gesture label, or None if below threshold.
  - `probs` (dict): Dictionary mapping gesture labels to probabilities for all classes.

**Note:**

- The model predicts only one gesture per image.
- No hand keypoints or bounding boxes are returned.

## MotionDetection API

### Class: `FrameDifferenceMotionDetector`

Detects moving regions in a video stream by comparing consecutive frames. Returns a motion mask, bounding boxes, contours, and motion vectors for each detected region.

#### Motion Vector Visualization

In motion mode, for each detected moving region, the average motion direction is calculated using optical flow and visualized as an arrow (motion vector) on the video stream. This helps to intuitively understand the direction and strength of movement in real time.

#### Initialization

```python
detector = FrameDifferenceMotionDetector(threshold=30, min_area=500)
```

- `threshold` (int): Pixel intensity threshold for motion detection.
- `min_area` (int): Minimum area (in pixels) for a region to be considered as motion.

#### Method: `detect`

```python
result = detector.detect(frame)
```

- `frame` (np.ndarray): Input video frame (BGR).
- Returns:
  - `motion_mask` (np.ndarray): Binary mask of detected motion.
  - `motion_bbox` (list): List of bounding boxes for moving regions.
  - `motion_contours` (list): List of contours for moving regions.
  - `motion_vectors` (list): List of motion vectors (center, direction) for each region.

#### MotionDetectionApp

The `motion_detection_app.py` demo visualizes moving regions and their motion vectors in real time. Press `1` to enable motion visualization mode (mask + arrows), `0` for normal video, and `ESC` to exit.

**Example usage:**

```bash
python examples/motion_detection_app.py
```

## Color Segmentation Demo

Segments an image into N dominant colors using KMeans clustering. This demo simplifies an image by reducing its color palette, which can be used for stylization or preprocessing.

**How to use:**

1. Set the input image path and number of colors in `examples/color_segmentation_demo.py`:

   ```python
   _img_path_in = "../assets/images/background.jpg"
   _img_path_out = "../assets/images/segmented_output.jpg"
   _n_colors = 5
   _segmenter = ColorSegmentation(image_path=_img_path_in, n_colors=_n_colors)
   segmented_img = _segmenter.segment()
   cv2.imwrite(_img_path_out, segmented_img)
   ```

2. Run the script:

   ```bash
   python examples/color_segmentation_demo.py
   ```

The output image will be saved to the specified path.

**Requirements:**

- scikit-learn
- numpy
- opencv-python

Install with:

```bash
pip install scikit-learn numpy opencv-python
```

---

## Hidden Area Effects

Anonymization effects for masking or obscuring specific regions in images. All effects support both mask-based and contour-based input.

### Available Effects

| Effect          | Description                              | Key Parameter         | Anonymization Level |
| --------------- | ---------------------------------------- | --------------------- | ------------------- |
| **Blur**        | Gaussian blur for soft anonymization     | `size` (31-101)       | ⭐⭐⭐              |
| **Mosaic**      | Pixelated blocks with average color      | `size` (20-50)        | ⭐⭐⭐⭐⭐          |
| **Solid Color** | Complete fill with single color          | `color` (BGR tuple)   | ⭐⭐⭐⭐⭐          |
| **Noise**       | Random color noise pattern               | `intensity` (0-255)   | ⭐⭐⭐⭐⭐          |
| **Bar**         | Alternating colored bars (TV censorship) | `bar_width`, `colors` | ⭐⭐⭐⭐⭐          |

### Usage Example

```python
from hidden_area_effects import BlurHiddenAreaEffect, MosaicHiddenAreaEffect
from background_effects.person_segmentation import PersonSegmentation

# Initialize
segmenter = PersonSegmentation()
effect = MosaicHiddenAreaEffect(size=30)

# Apply to person contours
contours = segmenter.get_contours(frame)
for contour in contours:
    frame = effect.apply_contour(frame, contour)

# Or apply with mask
mask = segmenter.get_binary_mask(frame)
frame = effect.apply(frame, mask)
```

### Demo

```bash
python examples/blur_face_demo.py
```

Switch effects by editing the import in `blur_face_demo.py`.

---

## License

See LICENSE file for details.
