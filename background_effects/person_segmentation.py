import cv2
import numpy as np
import logging
from typing import Optional

try:
    import mediapipe as mp
except ImportError:
    mp = None


class PersonSegmentation:
    def __init__(self, 
                 model_selection: int = 1,
                 logger: Optional[logging.Logger] = None):
        if mp is None:
            raise ImportError("mediapipe not installed. Install: pip install mediapipe")
        
        self.model_selection = model_selection
        self.logger = logger
        
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        
        if self.logger:
            model_type = "General" if model_selection == 0 else "Landscape"
            self.logger.info(f"Person Segmentation initialized (model={model_type})")
    
    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentation.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            return mask
        
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    def get_binary_mask(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        mask = self.get_mask(frame)
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        return binary_mask
    
    def get_contours(self, frame: np.ndarray, threshold: float = 0.5):
        binary_mask = self.get_binary_mask(frame, threshold)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def visualize_contours(self, frame: np.ndarray, contour_color=(0, 255, 0), 
                           thickness: int = 2, threshold: float = 0.5) -> np.ndarray:
        result = frame.copy()
        contours = self.get_contours(frame, threshold)
        cv2.drawContours(result, contours, -1, contour_color, thickness)
        return result
    
    def visualize_mask(self, frame: np.ndarray, mask_color=(0, 255, 0), alpha: float = 0.5) -> np.ndarray:
        mask = self.get_mask(frame)
        
        mask_3d = np.stack([mask] * 3, axis=-1)
        
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[:, :] = mask_color
        
        overlay = (colored_mask * mask_3d * alpha + frame * (1 - mask_3d * alpha)).astype(np.uint8)
        
        return overlay
    
    def __del__(self):
        if hasattr(self, 'segmentation'):
            self.segmentation.close()
