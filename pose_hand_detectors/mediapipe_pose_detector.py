import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_keypoint_detector import BaseKeypointDetector

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipePoseDetector(BaseKeypointDetector):
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
        (27, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]
    
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner',
        'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left',
        'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel',
        'right_heel', 'left_foot_index', 'right_foot_index'
    ]
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        if mp is None:
            raise ImportError("mediapipe not installed. Install: pip install mediapipe")
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logger
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        if self.logger:
            self.logger.info(f"MediaPipe Pose Detector initialized (complexity={model_complexity})")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        detections = []
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            
            keypoints = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints.append({
                    'x': landmark.x * w,
                    'y': landmark.y * h,
                    'confidence': landmark.visibility,
                    'name': self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f'point_{idx}'
                })
            
            detections.append({
                'type': 'pose',
                'keypoints': keypoints,
                'connections': self.POSE_CONNECTIONS
            })
        
        return detections

    def get_detector_name(self) -> str:
        return "MediaPipe Pose Detector"
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()
