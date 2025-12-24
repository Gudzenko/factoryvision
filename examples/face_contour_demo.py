from utils import SourceFactory, SourceType, WindowDisplay
from background_effects import PersonSegmentation
from style_effects import AdaptiveThresholdEffect
import cv2
import numpy as np
import logging


class FaceContourApp:
    def __init__(self, window_name="Face Contour Demo", is_flip=True, logger=None):
        self.window_name = window_name
        self.is_flip = is_flip
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        self.segmentation = PersonSegmentation(model_selection=1, logger=self.logger)
        self.contour_effect = AdaptiveThresholdEffect(
            block_size=9,
            C=3,
            method=cv2.ADAPTIVE_THRESH_MEAN_C,
            invert=False,
            logger=self.logger,
        )
        
        self.display = WindowDisplay(window_name)
        
        if self.logger:
            self.logger.info(f"Face Contour App initialized")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        denoised = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        mask = self.segmentation.get_binary_mask(frame, threshold=0.5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contour_result = self.contour_effect.apply(denoised)
        
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        white_bg = np.ones_like(frame) * 255
        result = np.where(mask_3ch > 0, contour_result, white_bg)
        
        return result.astype(np.uint8)
    
    def run(self):
        source = SourceFactory.create(SourceType.CAMERA, logger=self.logger)
        
        self.logger.info("Press ESC to exit")
        self.logger.info(f"Effect: Clean face contours with person segmentation")
        
        try:
            while True:
                frame = source.read()
                if frame is None:
                    continue
                
                if self.is_flip:
                    frame = cv2.flip(frame, 1)
                
                processed_frame = self.process_frame(frame)
                self.display.show_frame(processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
        
        finally:
            source.release()
            self.display.close()
            self.logger.info("Application stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('FaceContourDemo')
    
    app = FaceContourApp(is_flip=True, logger=_logger)
    app.run()
