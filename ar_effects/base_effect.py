from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any


class BaseAREffect(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        pass
