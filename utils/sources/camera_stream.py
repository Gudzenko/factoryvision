import cv2
import numpy as np
import logging
from typing import Optional
from .base_source import BaseSource


class CameraStream(BaseSource):
    def __init__(self, camera_id=0, logger: Optional[logging.Logger] = None):
        self.camera_id = camera_id
        self.logger = logger
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with ID {camera_id}")
        
        if self.logger:
            self.logger.info(f"Camera opened: ID {camera_id}")

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
        if self.logger:
            self.logger.info(f"Camera released: ID {self.camera_id}")
