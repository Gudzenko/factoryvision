import abc
import numpy as np


class BaseHiddenAreaEffect(abc.ABC):
    @abc.abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass

    def apply_contour(self, image: np.ndarray, contour, fill_value: int = 255) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if isinstance(contour, list):
            contour = np.array(contour, dtype=np.int32)
        if contour.ndim == 2:
            contour = contour.reshape((-1, 1, 2))
        import cv2
        cv2.fillPoly(mask, [contour], fill_value)
        return self.apply(image, mask)
