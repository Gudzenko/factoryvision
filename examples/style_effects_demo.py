from utils import SourceFactory, SourceType, WindowDisplay
from style_effects import (BaseStyleEffect, CannyEdgeEffect, PencilSketchEffect, 
                           CartoonEffect, AdaptiveThresholdEffect, OilPaintingEffect)
from typing import List
import cv2
import logging


class StyleEffectsDemoApp:
    def __init__(self, effects: List[BaseStyleEffect], window_name="Style Effects Demo", 
                 is_flip=True, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        
        self.effects = effects
        self.current_effect_index = 0
        
        self.is_flip = is_flip
        self.logger = logger

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        if 0 < self.current_effect_index <= len(self.effects):
            effect = self.effects[self.current_effect_index - 1]
            result = effect.apply(frame)
            
            effect_name = effect.get_effect_name()
            cv2.putText(result, f"Effect: {effect_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            result = frame.copy()
            cv2.putText(result, "Effect: Original", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result

    def run(self):
        if self.logger:
            self.logger.info("Starting Style Effects Demo...")
            self.logger.info(f"Available effects: {len(self.effects)}")
            for idx, effect in enumerate(self.effects):
                self.logger.info(f"  [{idx+1}] {effect.get_effect_name()}")
            self.logger.info("Press '1-9' to switch effects, '0' for original, ESC to exit")
        
        while True:
            frame = self.source.read()
            if frame is None:
                break

            result = self.process_frame(frame)
            self.window.show_frame(result)

            key = self.window.wait_key()
            
            if ord('0') <= key <= ord('9'):
                self.current_effect_index = key - ord('0')
                if self.logger:
                    if self.current_effect_index == 0:
                        self.logger.info("Switched to: Original")
                    elif self.current_effect_index <= len(self.effects):
                        effect_name = self.effects[self.current_effect_index - 1].get_effect_name()
                        self.logger.info(f"Switched to: {effect_name}")
            
            if self.window.should_close(key):
                break

        self.source.release()
        self.window.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('StyleEffectsDemo')

    _effects = [
        CannyEdgeEffect(threshold1=100, threshold2=150, invert=True, logger=_logger),
        PencilSketchEffect(sigma_s=75, sigma_r=0.07, shade_factor=0.1, logger=_logger),
        CartoonEffect(d=5, sigma_color=30, sigma_space=30, logger=_logger),
        AdaptiveThresholdEffect(block_size=11, C=2, method=cv2.ADAPTIVE_THRESH_MEAN_C, invert=False, logger=_logger),
        OilPaintingEffect(size=7, dyn_ratio=1, logger=_logger),
    ]
    
    app = StyleEffectsDemoApp(effects=_effects, is_flip=True, logger=_logger)
    app.run()
