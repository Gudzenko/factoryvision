import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_effect import BaseAREffect


class GlassesEffect(BaseAREffect):
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    NOSE_BRIDGE = 168
    
    def __init__(self, image_path: str, scale_factor: float = 1.8, 
                 logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.scale_factor = scale_factor
        self.glasses_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if self.glasses_img is None:
            raise FileNotFoundError(f"Glasses image not found: {image_path}")
        
        if self.glasses_img.shape[2] != 4:
            raise ValueError("Glasses image must have alpha channel (RGBA)")
        
        if self.logger:
            self.logger.info(f"Glasses effect initialized with image: {image_path}")
    
    def apply(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        result_frame = frame.copy()
        
        for detection in detections:
            if detection.get('type') != 'face_mesh':
                continue
            
            keypoints = detection.get('keypoints', [])
            if len(keypoints) < 468:
                continue
            
            left_eye_outer = keypoints[self.LEFT_EYE_OUTER]
            left_eye_inner = keypoints[self.LEFT_EYE_INNER]
            right_eye_inner = keypoints[self.RIGHT_EYE_INNER]
            right_eye_outer = keypoints[self.RIGHT_EYE_OUTER]
            nose_bridge = keypoints[self.NOSE_BRIDGE]
            
            left_point = np.array([left_eye_outer['x'], left_eye_outer['y']])
            right_point = np.array([right_eye_outer['x'], right_eye_outer['y']])
            nose_point = np.array([nose_bridge['x'], nose_bridge['y']])
            
            eyes_center = (left_point + right_point) / 2
            eyes_distance = np.linalg.norm(right_point - left_point)
            
            angle = np.arctan2(right_point[1] - left_point[1], 
                               right_point[0] - left_point[0])
            angle_deg = -np.degrees(angle)
            
            glasses_width = int(eyes_distance * self.scale_factor)
            glasses_height = int(self.glasses_img.shape[0] * glasses_width / self.glasses_img.shape[1])
            
            resized_glasses = cv2.resize(self.glasses_img, (glasses_width, glasses_height))
            
            cos_a = abs(np.cos(np.radians(angle_deg)))
            sin_a = abs(np.sin(np.radians(angle_deg)))
            new_width = int(glasses_width * cos_a + glasses_height * sin_a)
            new_height = int(glasses_width * sin_a + glasses_height * cos_a)
            
            M = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), angle_deg, 1.0)
            M[0, 2] += (new_width - glasses_width) / 2
            M[1, 2] += (new_height - glasses_height) / 2
            
            rotated_glasses = cv2.warpAffine(resized_glasses, M, (new_width, new_height))
            
            x_offset = int(eyes_center[0] - new_width // 2)
            y_offset = int(eyes_center[1] - new_height // 2)
            
            result_frame = self._overlay_image(result_frame, rotated_glasses, x_offset, y_offset)
        
        return result_frame
    
    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray, 
                       x: int, y: int) -> np.ndarray:
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
            return background
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(bg_w, x + w)
        y2 = min(bg_h, y + h)
        
        overlay_x1 = x1 - x
        overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)
        
        overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        background_crop = background[y1:y2, x1:x2]
        
        if overlay_crop.shape[2] == 4:
            alpha = overlay_crop[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            overlay_rgb = overlay_crop[:, :, :3]
            blended = overlay_rgb * alpha + background_crop * (1 - alpha)
            background[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            background[y1:y2, x1:x2] = overlay_crop
        
        return background
