import numpy as np
from .base_hidden_area_effect import BaseHiddenAreaEffect


class SolidColorHiddenAreaEffect(BaseHiddenAreaEffect):
    def __init__(self, color=(0, 0, 0)):
        self.color = np.array(color, dtype=np.uint8)

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        result = image.copy()
        result[mask_bool, :] = self.color
        return result
