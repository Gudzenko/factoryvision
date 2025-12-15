from abc import ABC, abstractmethod
import numpy as np


class BaseSource(ABC):
    @abstractmethod
    def read(self) -> np.ndarray:
        pass

    @abstractmethod
    def release(self):
        pass
