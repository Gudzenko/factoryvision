from utils.camera_stream import CameraStream
from utils.window_display import WindowDisplay
from utils.detection_drawer import DetectionDrawer
from face_body_detectors import DetectorFactory, DetectorType
import cv2
import logging


class FactoryVisionApp:
    def __init__(self, window_name="FactoryVision Demo", is_flip=False, logger=None):
        self.camera = CameraStream()
        self.window = WindowDisplay(window_name)
        
        self.detector = DetectorFactory.create(DetectorType.MEDIAPIPE, logger=logger, model_selection=1)
        self.drawer = DetectionDrawer(label_text="Face")
        self.is_flip = is_flip

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        detections = self.detector.detect(frame)
        frame = self.drawer.draw(frame, detections)
        
        return frame

    def run(self):
        while True:
            frame = self.camera.read()
            if frame is None:
                break

            frame = self.process_frame(frame)
            self.window.show_frame(frame)

            key = self.window.wait_key()
            if self.window.should_close(key):
                break

        self.camera.release()
        self.window.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('FactoryVision')

    app = FactoryVisionApp(is_flip=True, logger=_logger)
    app.run()
