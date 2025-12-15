import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class UniversalDrawer:
    def __init__(self, 
                 box_color: Tuple[int, int, int] = (0, 255, 0),
                 box_thickness: int = 2,
                 point_color: Tuple[int, int, int] = (0, 255, 0),
                 line_color: Tuple[int, int, int] = (255, 0, 0),
                 point_radius: int = 4,
                 line_thickness: int = 2,
                 confidence_threshold: float = 0.5,
                 label_font_scale: float = 0.6,
                 label_font_thickness: int = 2):
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.point_color = point_color
        self.line_color = line_color
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        self.confidence_threshold = confidence_threshold
        self.label_font_scale = label_font_scale
        self.label_font_thickness = label_font_thickness

    def _draw_boxes(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]], 
                     label_text: str = "Object") -> np.ndarray:
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
            
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.label_font_scale, self.label_font_thickness)
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), self.box_color, -1)
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, 
                       (0, 0, 0), self.label_font_thickness)
        
        return frame

    def _draw_keypoints(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for detection in detections:
            keypoints = detection.get('keypoints', [])
            connections = detection.get('connections', [])
            
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    kp1 = keypoints[idx1]
                    kp2 = keypoints[idx2]
                    
                    if (kp1['confidence'] >= self.confidence_threshold and 
                        kp2['confidence'] >= self.confidence_threshold):
                        pt1 = (int(kp1['x']), int(kp1['y']))
                        pt2 = (int(kp2['x']), int(kp2['y']))
                        cv2.line(frame, pt1, pt2, self.line_color, self.line_thickness)
            
            for keypoint in keypoints:
                if keypoint['confidence'] >= self.confidence_threshold:
                    center = (int(keypoint['x']), int(keypoint['y']))
                    cv2.circle(frame, center, self.point_radius, self.point_color, -1)
        
        return frame
    
    def draw_all(self, frame: np.ndarray, box_detections: List[Tuple[int, int, int, int]] = None,
                 keypoint_detections: List[Dict[str, Any]] = None, label_text: str = "Object") -> np.ndarray:
        result_frame = frame.copy()
        
        if box_detections:
            result_frame = self._draw_boxes(result_frame, box_detections, label_text)
        
        if keypoint_detections:
            result_frame = self._draw_keypoints(result_frame, keypoint_detections)
        
        return result_frame
