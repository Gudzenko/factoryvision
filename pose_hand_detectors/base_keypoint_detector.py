from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseKeypointDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns list of detections:
        {
            'type': 'pose' or 'hand',
            'keypoints': [{'x': float, 'y': float, 'confidence': float, 'name': str}, ...],
            'connections': [(idx1, idx2), ...]
        }
        """
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        pass
