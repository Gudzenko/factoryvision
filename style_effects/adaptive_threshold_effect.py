import cv2
import numpy as np
import logging
from typing import Optional
from .base_style_effect import BaseStyleEffect


class AdaptiveThresholdEffect(BaseStyleEffect):
    def __init__(self, block_size: int = 11, C: int = 2, 
                 method: int = cv2.ADAPTIVE_THRESH_MEAN_C,
                 invert: bool = False,
                 logger: Optional[logging.Logger] = None):
        if block_size % 2 == 0:
            block_size += 1
        
        self.block_size = block_size
        self.C = C
        self.method = method
        self.invert = invert
        self.logger = logger
        
        method_name = "MEAN" if method == cv2.ADAPTIVE_THRESH_MEAN_C else "GAUSSIAN"
        if self.logger:
            self.logger.info(f"Adaptive Threshold Effect initialized (block_size={block_size}, "
                           f"C={C}, method={method_name}, invert={invert})")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray_blur = cv2.medianBlur(gray, 5)
        
        thresh_type = cv2.THRESH_BINARY_INV if self.invert else cv2.THRESH_BINARY
        
        result_gray = cv2.adaptiveThreshold(
            gray_blur, 
            255, 
            self.method, 
            thresh_type, 
            self.block_size, 
            self.C
        )
        
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def get_effect_name(self) -> str:
        method_name = "MEAN" if self.method == cv2.ADAPTIVE_THRESH_MEAN_C else "GAUSS"
        return f"Adaptive Threshold (bs={self.block_size}, C={self.C}, {method_name})"
