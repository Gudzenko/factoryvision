from abc import ABC, abstractmethod
import numpy as np

class BaseHandGestureRecognizer(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> str:
        pass
