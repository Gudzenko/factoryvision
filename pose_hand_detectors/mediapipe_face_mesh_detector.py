import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .base_keypoint_detector import BaseKeypointDetector

try:
    import mediapipe as mp
except ImportError:
    mp = None


class MediaPipeFaceMeshDetector(BaseKeypointDetector):
    def __init__(self, 
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        if mp is None:
            raise ImportError("mediapipe not installed. Install: pip install mediapipe")
        
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logger
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.connections = list(self.mp_face_mesh.FACEMESH_CONTOURS)
        
        if self.logger:
            self.logger.info(f"MediaPipe Face Mesh Detector initialized (max_faces={max_num_faces}, refine={refine_landmarks})")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        detections = []
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            
            for face_landmarks in results.multi_face_landmarks:
                keypoints = []
                for idx, landmark in enumerate(face_landmarks.landmark):
                    keypoints.append({
                        'x': landmark.x * w,
                        'y': landmark.y * h,
                        'confidence': 1.0,
                        'name': f'face_point_{idx}'
                    })
                
                detections.append({
                    'type': 'face_mesh',
                    'keypoints': keypoints,
                    'connections': self.connections
                })
        
        return detections

    def get_detector_name(self) -> str:
        return "MediaPipe Face Mesh Detector"
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
