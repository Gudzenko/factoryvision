from utils.sources import SourceFactory, SourceType
from utils.window_display import WindowDisplay
from hidden_area_effects import MosaicHiddenAreaEffect
import numpy as np
from background_effects.person_segmentation import PersonSegmentation
import cv2


class BlurFaceDemo:
    def __init__(self, window_name="Blur Face Demo", is_flip=True, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        self.segmenter = PersonSegmentation(logger=logger)
        self.is_flip = is_flip
        self.logger = logger

    def process_frame(self, frame):
        if self.is_flip:
            frame = np.ascontiguousarray(np.flip(frame, axis=1))
        result = frame.copy()
        contours = self.segmenter.get_contours(frame)
        for contour in contours:
            result = MosaicHiddenAreaEffect(size=20).apply_contour(image=result, contour=contour)
            cv2.polylines(result, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
        return result

    def run(self):
        if self.logger:
            self.logger.info("Starting Blur Face Demo...")
        while True:
            frame = self.source.read()
            if frame is None:
                break
            frame = self.process_frame(frame)
            self.window.show_frame(frame)
            key = self.window.wait_key()
            if self.window.should_close(key):
                break
        self.source.release()
        self.window.close()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger("BlurFaceDemo")
    BlurFaceDemo(logger=_logger).run()
