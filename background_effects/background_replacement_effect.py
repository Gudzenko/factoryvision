import cv2
import numpy as np
import logging
from typing import Optional


class BackgroundReplacementEffect:
    def __init__(self, background_image_path: str, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.background_img = cv2.imread(background_image_path)
        
        if self.background_img is None:
            raise FileNotFoundError(f"Background image not found: {background_image_path}")
        
        if self.logger:
            self.logger.info(f"Background replacement initialized with image: {background_image_path}")
    
    def apply(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        background_resized = cv2.resize(self.background_img, (w, h))
        
        if mask.dtype != np.float32 and mask.dtype != np.float64:
            mask_normalized = mask.astype(np.float32) / 255.0
        else:
            mask_normalized = mask
        
        mask_3d = np.stack([mask_normalized] * 3, axis=-1)
        
        result = (frame * mask_3d + background_resized * (1 - mask_3d)).astype(np.uint8)
        return result
