import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_keypoint_detector import BaseKeypointDetector

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipeHandsDetector(BaseKeypointDetector):
    HAND_CONNECTIONS = [
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    LANDMARK_NAMES = [
        'wrist',
        'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
        'index_mcp', 'index_pip', 'index_dip', 'index_tip',
        'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
        'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
        'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
    ]
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        if mp is None:
            raise ImportError("mediapipe not installed. Install: pip install mediapipe")
        
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logger
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        if self.logger:
            self.logger.info(f"MediaPipe Hands Detector initialized (max_hands={max_num_hands}, complexity={model_complexity})")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detections = []
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                keypoints = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    keypoints.append({
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'confidence': 1.0,
                        'name': self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f'point_{idx}'
                    })
                
                handedness = "Unknown"
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                
                detections.append({
                    'type': 'hand',
                    'handedness': handedness,
                    'keypoints': keypoints,
                    'connections': self.HAND_CONNECTIONS
                })
        
        return detections

    def get_detector_name(self) -> str:
        return "MediaPipe Hands Detector"
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()
