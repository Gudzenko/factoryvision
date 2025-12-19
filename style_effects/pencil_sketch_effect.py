import cv2
import numpy as np
import logging
from typing import Optional
from .base_style_effect import BaseStyleEffect


class PencilSketchEffect(BaseStyleEffect):
    def __init__(self, sigma_s: int = 60, sigma_r: float = 0.07, 
                 shade_factor: float = 0.05, logger: Optional[logging.Logger] = None):
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.shade_factor = shade_factor
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Pencil Sketch Effect initialized (sigma_s={sigma_s}, sigma_r={sigma_r}, shade_factor={shade_factor})")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray_sketch, color_sketch = cv2.pencilSketch(
            frame, 
            sigma_s=self.sigma_s, 
            sigma_r=self.sigma_r, 
            shade_factor=self.shade_factor
        )
        
        result = cv2.cvtColor(gray_sketch, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def get_effect_name(self) -> str:
        return f"Pencil Sketch (s={self.sigma_s}, r={self.sigma_r:.2f})"
