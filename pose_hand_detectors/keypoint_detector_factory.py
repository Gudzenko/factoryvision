import logging
from typing import Optional
from enum import Enum

from .mediapipe_pose_detector import MediaPipePoseDetector
from .mediapipe_hands_detector import MediaPipeHandsDetector
from .mediapipe_face_mesh_detector import MediaPipeFaceMeshDetector
from .yolo_pose_detector import YOLOPoseDetector
from .base_keypoint_detector import BaseKeypointDetector


class KeypointDetectorType(Enum):
    MEDIAPIPE_POSE = "mediapipe_pose"
    MEDIAPIPE_HANDS = "mediapipe_hands"
    MEDIAPIPE_FACE_MESH = "mediapipe_face_mesh"
    YOLO_POSE = "yolo_pose"


class KeypointDetectorFactory:
    @staticmethod
    def create(detector_type: KeypointDetectorType, 
               logger: Optional[logging.Logger] = None, 
               **kwargs) -> BaseKeypointDetector:
        if detector_type == KeypointDetectorType.MEDIAPIPE_POSE:
            return MediaPipePoseDetector(logger=logger, **kwargs)
        
        elif detector_type == KeypointDetectorType.MEDIAPIPE_HANDS:
            return MediaPipeHandsDetector(logger=logger, **kwargs)
        
        elif detector_type == KeypointDetectorType.MEDIAPIPE_FACE_MESH:
            return MediaPipeFaceMeshDetector(logger=logger, **kwargs)
        
        elif detector_type == KeypointDetectorType.YOLO_POSE:
            return YOLOPoseDetector(logger=logger, **kwargs)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
