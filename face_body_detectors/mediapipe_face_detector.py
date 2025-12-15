import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from .base_detector import BaseDetector

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipeFaceDetector(BaseDetector):
    def __init__(self, 
                 model_selection: int = 0,
                 min_detection_confidence: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        if mp is None:
            raise ImportError("mediapipe not installed. Install: pip install mediapipe")
        
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.logger = logger
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
        
        if self.logger:
            self.logger.info(f"MediaPipe Face Detector initialized (model={model_selection}, conf={min_detection_confidence})")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        boxes = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                boxes.append((x, y, width, height))
        
        return boxes

    def get_detector_name(self) -> str:
        return "MediaPipe Face Detector"
    
    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
