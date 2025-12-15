from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces or persons in the given frame.

        Args:
            frame: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of bounding boxes in format [(x, y, width, height), ...]
            where (x, y) is the top-left corner of the bounding box
        """
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        """
        Get the name of the detector.

        Returns:
            String with detector name
        """
        pass
