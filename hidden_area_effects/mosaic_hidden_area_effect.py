import cv2
import numpy as np
from .base_hidden_area_effect import BaseHiddenAreaEffect


class MosaicHiddenAreaEffect(BaseHiddenAreaEffect):
    def __init__(self, size: int = 20):
        self.size = size

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        result = image.copy()
        size = self.size
        for y in range(0, h, size):
            for x in range(0, w, size):
                y_end = min(y + size, h)
                x_end = min(x + size, w)
                block_mask = mask_bool[y:y_end, x:x_end]
                if block_mask.any():
                    block = image[y:y_end, x:x_end]
                    masked_pixels = block[block_mask]
                    if len(masked_pixels) > 0:
                        mean_color = masked_pixels.mean(axis=0)
                        result[y:y_end, x:x_end][block_mask] = mean_color
        return result
