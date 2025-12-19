from abc import ABC, abstractmethod
import numpy as np


class BaseStyleEffect(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_effect_name(self) -> str:
        pass
