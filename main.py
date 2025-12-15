from utils import SourceFactory, SourceType, WindowDisplay, DetectionDrawer
from face_body_detectors import DetectorFactory, DetectorType
import cv2
import logging


class FactoryVisionApp:
    def __init__(self, window_name="FactoryVision Demo", is_flip=False, logger=None):
        self.source = SourceFactory.create(
            SourceType.VIDEO_FILE, 
            video_path="assets/videos/dance.mp4", 
            loop=True, 
            logger=logger,
            realtime=False,
        )
        self.window = WindowDisplay(window_name)
        
        self.detector = DetectorFactory.create(DetectorType.YOLO, logger=logger, target_classes=[0])
        self.drawer = DetectionDrawer(label_text="Person")
        self.is_flip = is_flip

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        detections = self.detector.detect(frame)
        frame = self.drawer.draw(frame, detections)
        
        return frame

    def run(self):
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('FactoryVision')

    app = FactoryVisionApp(is_flip=False, logger=_logger)
    app.run()
