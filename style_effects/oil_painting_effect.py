import cv2
import numpy as np
import logging
from typing import Optional
from .base_style_effect import BaseStyleEffect


class OilPaintingEffect(BaseStyleEffect):
    def __init__(self, size: int = 7, dyn_ratio: int = 1,
                 logger: Optional[logging.Logger] = None):
        self.size = size
        self.dyn_ratio = dyn_ratio
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Oil Painting Effect initialized (size={size}, dynRatio={dyn_ratio})")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        try:
            result = cv2.xphoto.oilPainting(frame, self.size, self.dyn_ratio)
            return result
        except AttributeError:
            if self.logger:
                self.logger.warning("cv2.xphoto.oilPainting not available, using fallback method")
            return self._apply_fallback(frame)
    
    def _apply_fallback(self, frame: np.ndarray) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=90, sigmaSpace=90)
        
        levels = 8
        quantized = (smooth // (256 // levels)) * (256 // levels)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        result = cv2.medianBlur(quantized, 5)
        
        return result
    
    def get_effect_name(self) -> str:
        return f"Oil Painting (size={self.size}, dyn={self.dyn_ratio})"
