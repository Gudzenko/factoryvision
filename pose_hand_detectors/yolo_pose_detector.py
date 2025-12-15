import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_keypoint_detector import BaseKeypointDetector

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOPoseDetector(BaseKeypointDetector):
    POSE_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    LANDMARK_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, 
                 model_name: str = 'yolo11n-pose.pt',
                 confidence_threshold: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install: pip install ultralytics")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.logger = logger
        
        self.model = YOLO(model_name)
        
        if self.logger:
            self.logger.info(f"YOLO Pose Detector initialized (model={model_name})")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            if result.keypoints is None:
                continue
            
            for keypoints_data in result.keypoints:
                if keypoints_data.conf is None:
                    continue
                
                keypoints = []
                xy = keypoints_data.xy[0].cpu().numpy()
                conf = keypoints_data.conf[0].cpu().numpy()
                
                for idx, (point, confidence) in enumerate(zip(xy, conf)):
                    keypoints.append({
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'confidence': float(confidence),
                        'name': self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f'point_{idx}'
                    })
                
                detections.append({
                    'type': 'pose',
                    'keypoints': keypoints,
                    'connections': self.POSE_CONNECTIONS
                })
        
        return detections

    def get_detector_name(self) -> str:
        return f"YOLO Pose Detector ({self.model_name})"
