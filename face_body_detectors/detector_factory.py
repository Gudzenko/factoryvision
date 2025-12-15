import logging
from typing import Optional
from enum import Enum

from .opencv_face_detector import OpenCVFaceDetector
from .dnn_face_detector import DNNFaceDetector
from .yolo_detector import YOLODetector
from .mediapipe_face_detector import MediaPipeFaceDetector
from .base_detector import BaseDetector


class DetectorType(Enum):
    HAAR_CASCADE = "haar"
    DNN_FACE = "dnn"
    YOLO = "yolo"
    MEDIAPIPE = "mediapipe"


class DetectorFactory:
    @staticmethod
    def create(detector_type: DetectorType, logger: Optional[logging.Logger] = None, **kwargs) -> BaseDetector:
        if detector_type == DetectorType.HAAR_CASCADE:
            return OpenCVFaceDetector(logger=logger, **kwargs)
        
        elif detector_type == DetectorType.DNN_FACE:
            return DNNFaceDetector(logger=logger, **kwargs)
        
        elif detector_type == DetectorType.YOLO:
            return YOLODetector(logger=logger, **kwargs)
        
        elif detector_type == DetectorType.MEDIAPIPE:
            return MediaPipeFaceDetector(logger=logger, **kwargs)
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
