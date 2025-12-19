import cv2
import numpy as np
import logging
from typing import Optional
from .base_style_effect import BaseStyleEffect


class CartoonEffect(BaseStyleEffect):
    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75,
                 logger: Optional[logging.Logger] = None):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"Cartoon Effect initialized (d={d}, sigma_color={sigma_color}, "
                           f"sigma_space={sigma_space})")
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, d=self.d, sigmaColor=self.sigma_color, 
                                     sigmaSpace=self.sigma_space)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray_blur, 255, 
                                      cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 2)
        
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        result = cv2.bitwise_and(smooth, edges_colored)
        return result
    
    def get_effect_name(self) -> str:
        return f"Cartoon (d={self.d}, sigma={self.sigma_color})"
