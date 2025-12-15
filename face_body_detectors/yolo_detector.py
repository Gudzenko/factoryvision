import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from .base_detector import BaseDetector

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLODetector(BaseDetector):
    def __init__(self, 
                 model_name: str = "yolo11n.pt",
                 confidence_threshold: float = 0.5,
                 target_classes: List[int] = None,
                 logger: Optional[logging.Logger] = None):
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes or [0]
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Loading YOLO model: {model_name}")
        
        self.model = YOLO(model_name)
        
        if self.logger:
            self.logger.info(f"YOLO Detector initialized (model={model_name}, conf={confidence_threshold})")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    boxes.append((x, y, w, h))
        
        return boxes

    def get_detector_name(self) -> str:
        return "YOLO Detector"
