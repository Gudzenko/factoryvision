import cv2
import numpy as np
from typing import List, Tuple


class DetectionDrawer:
    def __init__(self, 
                 box_color: Tuple[int, int, int] = (0, 255, 0),
                 box_thickness: int = 2,
                 label_text: str = "Face",
                 font_scale: float = 0.6,
                 font_thickness: int = 2):
        self.box_color = box_color
        self.box_thickness = box_thickness
        self.label_text = label_text
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        result_frame = frame.copy()
        
        for (x, y, w, h) in detections:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), self.box_color, self.box_thickness)
            
            label_size, _ = cv2.getTextSize(self.label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.font_scale, self.font_thickness)
            cv2.rectangle(result_frame, (x, y - label_size[1] - 10), 
                          (x + label_size[0], y), self.box_color, -1)
            cv2.putText(result_frame, self.label_text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), self.font_thickness)
        
        return result_frame
