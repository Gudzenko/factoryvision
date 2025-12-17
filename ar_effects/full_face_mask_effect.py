import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_effect import BaseAREffect


class FullFaceMaskEffect(BaseAREffect):
    FOREHEAD_TOP = 10
    CHIN_BOTTOM = 152
    LEFT_TEMPLE = 234
    RIGHT_TEMPLE = 454
    
    def __init__(self, image_path: str, scale_factor: float = 2.5, 
                 x_offset: float = 0.0, y_offset: float = 0.0, 
                 logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.scale_factor = scale_factor
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.mask_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if self.mask_img is None:
            raise FileNotFoundError(f"Mask image not found: {image_path}")
        
        if self.mask_img.shape[2] != 4:
            raise ValueError("Mask image must have alpha channel (RGBA)")
        
        if self.logger:
            self.logger.info(f"Full face mask effect initialized with image: {image_path}")
    
    def apply(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        result_frame = frame.copy()
        
        for detection in detections:
            if detection.get('type') != 'face_mesh':
                continue
            
            keypoints = detection.get('keypoints', [])
            if len(keypoints) < 468:
                continue
            
            forehead = keypoints[self.FOREHEAD_TOP]
            chin = keypoints[self.CHIN_BOTTOM]
            left_temple = keypoints[self.LEFT_TEMPLE]
            right_temple = keypoints[self.RIGHT_TEMPLE]
            
            forehead_pt = np.array([forehead['x'], forehead['y']])
            chin_pt = np.array([chin['x'], chin['y']])
            left_pt = np.array([left_temple['x'], left_temple['y']])
            right_pt = np.array([right_temple['x'], right_temple['y']])
            
            face_width = np.linalg.norm(right_pt - left_pt)
            face_height = np.linalg.norm(chin_pt - forehead_pt)
            face_center = (forehead_pt + chin_pt) / 2
            
            angle = np.arctan2(right_pt[1] - left_pt[1], 
                              right_pt[0] - left_pt[0])
            angle_deg = -np.degrees(angle)
            
            mask_width = int(face_width * self.scale_factor)
            mask_height = int(self.mask_img.shape[0] * mask_width / self.mask_img.shape[1])
            
            resized_mask = cv2.resize(self.mask_img, (mask_width, mask_height))
            
            cos_a = abs(np.cos(np.radians(angle_deg)))
            sin_a = abs(np.sin(np.radians(angle_deg)))
            new_width = int(mask_width * cos_a + mask_height * sin_a)
            new_height = int(mask_width * sin_a + mask_height * cos_a)
            
            M = cv2.getRotationMatrix2D((mask_width // 2, mask_height // 2), angle_deg, 1.0)
            M[0, 2] += (new_width - mask_width) / 2
            M[1, 2] += (new_height - mask_height) / 2
            
            rotated_mask = cv2.warpAffine(resized_mask, M, (new_width, new_height))
            
            perpendicular_angle = angle - np.pi / 2
            
            y_offset_distance = face_height * self.y_offset
            offset_y_x = y_offset_distance * np.cos(perpendicular_angle)
            offset_y_y = y_offset_distance * np.sin(perpendicular_angle)
            
            x_offset_distance = face_width * self.x_offset
            offset_x_x = x_offset_distance * np.cos(angle)
            offset_x_y = x_offset_distance * np.sin(angle)
            
            mask_position_x = face_center[0] + offset_y_x + offset_x_x
            mask_position_y = face_center[1] + offset_y_y + offset_x_y
            
            x_offset = int(mask_position_x - new_width // 2)
            y_offset = int(mask_position_y - new_height // 2)
            
            result_frame = self._overlay_image(result_frame, rotated_mask, x_offset, y_offset)
        
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
