import numpy as np
from .base_hidden_area_effect import BaseHiddenAreaEffect


class BarHiddenAreaEffect(BaseHiddenAreaEffect):
    def __init__(self, bar_width: int = 10, orientation: str = 'horizontal', colors=None):
        self.bar_width = max(1, bar_width)
        self.orientation = orientation
        if colors is None:
            colors = [(0, 0, 0), (255, 255, 255)]
        self.colors = [np.array(c, dtype=np.uint8) for c in colors]

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        result = image.copy()
        h, w = image.shape[:2]
        
        if self.orientation == 'horizontal':
            bar_index = 0
            for y in range(0, h, self.bar_width):
                y_end = min(y + self.bar_width, h)
                bar_mask = mask_bool[y:y_end, :]
                color = self.colors[bar_index % len(self.colors)]
                result[y:y_end, :][bar_mask] = color
                bar_index += 1
        elif self.orientation == 'vertical':
            bar_index = 0
            for x in range(0, w, self.bar_width):
                x_end = min(x + self.bar_width, w)
                bar_mask = mask_bool[:, x:x_end]
                color = self.colors[bar_index % len(self.colors)]
                result[:, x:x_end][bar_mask] = color
                bar_index += 1
        else:
            raise ValueError(f"Unknown orientation: {self.orientation}. Use 'horizontal' or 'vertical'.")
        
        return result
