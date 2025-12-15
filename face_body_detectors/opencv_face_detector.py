import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from .base_detector import BaseDetector


class OpenCVFaceDetector(BaseDetector):
    def __init__(self, 
                 scale_factor: float = 1.1, 
                 min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (30, 30),
                 cascade_file: str = None,
                 logger: Optional[logging.Logger] = None):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.logger = logger
        
        if cascade_file:
            self.face_cascade = cv2.CascadeClassifier(cascade_file)
        else:
            self.face_cascade = cv2.CascadeClassifier(
                f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
            )
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
        
        if self.logger:
            self.logger.info("OpenCV Haar Cascade Face Detector initialized")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return [tuple(face) for face in faces]

    def get_detector_name(self) -> str:
        return "OpenCV Haar Cascade Face Detector"
