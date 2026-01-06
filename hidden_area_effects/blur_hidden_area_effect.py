import cv2
import numpy as np
from .base_hidden_area_effect import BaseHiddenAreaEffect


class BlurHiddenAreaEffect(BaseHiddenAreaEffect):
    def __init__(self, size: int = 31):
        if size <= 0:
            size = 31
        if size % 2 == 0:
            size += 1
        self.size = size

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        result = image.copy()
        blurred = cv2.GaussianBlur(image, (self.size, self.size), 0)
        result[mask_bool, :] = blurred[mask_bool, :]
        return result

