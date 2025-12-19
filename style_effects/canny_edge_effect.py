import cv2
import numpy as np
import logging
from typing import Optional
from .base_style_effect import BaseStyleEffect


class CannyEdgeEffect(BaseStyleEffect):
    def __init__(self, threshold1: int = 50, threshold2: int = 150, 
                 invert: bool = True, logger: Optional[logging.Logger] = None):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.invert = invert
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Canny Edge Effect initialized (threshold1={threshold1}, threshold2={threshold2}, invert={invert})")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        
        if self.invert:
            edges = 255 - edges
        
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def get_effect_name(self) -> str:
        return f"Canny Edge (t1={self.threshold1}, t2={self.threshold2})"
