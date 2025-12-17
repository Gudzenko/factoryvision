import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_effect import BaseAREffect


class FrameEffect(BaseAREffect):
    def __init__(self, image_path: str, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.frame_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if self.frame_img is None:
            raise FileNotFoundError(f"Frame image not found: {image_path}")
        
        if self.frame_img.shape[2] != 4:
            raise ValueError("Frame image must have alpha channel (RGBA)")
        
        if self.logger:
            self.logger.info(f"Frame effect initialized with image: {image_path}")
    
    def apply(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        h, w = frame.shape[:2]
        
        resized_frame_img = cv2.resize(self.frame_img, (w, h))
        
        result_frame = self._overlay_image(frame, resized_frame_img)
        
        return result_frame
    
    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            overlay_rgb = overlay[:, :, :3]
            blended = overlay_rgb * alpha + background * (1 - alpha)
            return blended.astype(np.uint8)
        else:
            return overlay
