# FactoryVision

Computer vision application for face and person detection using multiple detection methods.

## Features

- Multiple camera support
- Real-time detection and visualization
- Pluggable detector architecture
- Four different detection methods

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

## License

See LICENSE file for details.
