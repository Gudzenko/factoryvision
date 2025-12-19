from utils import SourceFactory, SourceType, WindowDisplay
from background_effects import PersonSegmentation, BackgroundReplacementEffect
import cv2
import logging


class SegmentationDemoApp:
    def __init__(self, background_path: str, window_name="Segmentation Demo", is_flip=True, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        
        self.segmentation = PersonSegmentation(model_selection=1, logger=logger)
        self.bg_replacement = BackgroundReplacementEffect(background_path, logger=logger)
        
        self.is_flip = is_flip
        self.logger = logger
        self.mode = 0

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        if self.mode == 0:
            mask = self.segmentation.get_mask(frame)
            result = (mask * 255).astype('uint8')
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif self.mode == 1:
            result = self.segmentation.visualize_mask(frame=frame, mask_color=(0, 255, 0), alpha=0.5)
        elif self.mode == 2:
            binary_mask = self.segmentation.get_binary_mask(frame=frame, threshold=0.7)
            result = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        elif self.mode == 3:
            result = self.segmentation.visualize_contours(
                frame=frame,
                contour_color=(0, 255, 0),
                thickness=2,
                threshold=0.7,
            )
        elif self.mode == 4:
            mask = self.segmentation.get_mask(frame)
            result = self.bg_replacement.apply(frame, mask)
        else:
            result = frame
        
        return result

    def run(self):
        if self.logger:
            self.logger.info("Starting Segmentation Demo...")
            self.logger.info("Press '1' - grayscale mask, '2' - colored overlay, '3' - binary mask, "
                             "'4' - contours, '5' - background replacement, '0' - original")
        
        while True:
            frame = self.source.read()
            if frame is None:
                break

            result = self.process_frame(frame)
            self.window.show_frame(result)

            key = self.window.wait_key()
            
            if key == ord('1'):
                self.mode = 0
            elif key == ord('2'):
                self.mode = 1
            elif key == ord('3'):
                self.mode = 2
            elif key == ord('4'):
                self.mode = 3
            elif key == ord('5'):
                self.mode = 4
            elif key == ord('0'):
                self.mode = 5
            
            if self.window.should_close(key):
                break

        self.source.release()
        self.window.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('SegmentationDemo')

    BACKGROUND_PATH = "assets/images/background.jpg"
    
    app = SegmentationDemoApp(background_path=BACKGROUND_PATH, is_flip=True, logger=_logger)
    app.run()
