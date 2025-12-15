import cv2
import numpy as np
import os
import urllib.request
import logging
from typing import List, Tuple, Optional
from .base_detector import BaseDetector


class DNNFaceDetector(BaseDetector):
    PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    def __init__(self, 
                 prototxt_path: str = None,
                 model_path: str = None,
                 confidence_threshold: float = 0.5,
                 models_dir: str = "models",
                 logger: Optional[logging.Logger] = None):
        self.confidence_threshold = confidence_threshold
        self.logger = logger
        
        if not prototxt_path:
            prototxt_path = os.path.join(models_dir, "deploy.prototxt")
        if not model_path:
            model_path = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        self._ensure_models_exist(prototxt_path, model_path, models_dir)
        
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
        if self.net.empty():
            raise ValueError("Failed to load DNN model")
    
    def _ensure_models_exist(self, prototxt_path: str, model_path: str, models_dir: str):
        os.makedirs(models_dir, exist_ok=True)
        
        if not os.path.exists(prototxt_path):
            if self.logger:
                self.logger.info(f"Downloading {prototxt_path}...")
            urllib.request.urlretrieve(self.PROTOTXT_URL, prototxt_path)
            if self.logger:
                self.logger.info(f"Downloaded {prototxt_path}")
        
        if not os.path.exists(model_path):
            if self.logger:
                self.logger.info(f"Downloading {model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            if self.logger:
                self.logger.info(f"Downloaded {model_path}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                x = max(0, x1)
                y = max(0, y1)
                width = min(x2 - x1, w - x)
                height = min(y2 - y1, h - y)
                
                boxes.append((x, y, width, height))
        
        return boxes

    def get_detector_name(self) -> str:
        return "OpenCV DNN Face Detector"
