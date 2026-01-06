import cv2
import numpy as np
from .base_hidden_area_effect import BaseHiddenAreaEffect


class NoiseHiddenAreaEffect(BaseHiddenAreaEffect):
    def __init__(self, intensity: int = 255):
        self.intensity = max(0, min(255, intensity))

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        result = image.copy()
        noise = np.random.randint(0, self.intensity + 1, size=image.shape, dtype=np.uint8)
        result[mask_bool, :] = noise[mask_bool, :]
        return result
