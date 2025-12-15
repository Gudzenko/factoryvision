from utils import SourceFactory, SourceType, WindowDisplay, UniversalDrawer
from face_body_detectors import DetectorFactory, DetectorType
from pose_hand_detectors import KeypointDetectorFactory, KeypointDetectorType
import cv2
import logging


class FactoryVisionApp:
    def __init__(self, window_name="FactoryVision Demo", is_flip=False, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        
        self.face_detector = DetectorFactory.create(
            detector_type=DetectorType.YOLO,
            logger=logger,
            target_classes=[0],
        )
        self.keypoint_detector = KeypointDetectorFactory.create(
            detector_type=KeypointDetectorType.MEDIAPIPE_FACE_MESH,
            logger=logger,
        )
        self.drawer = UniversalDrawer()
        
        self.is_flip = is_flip

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        face_detections = self.face_detector.detect(frame)
        keypoint_detections = self.keypoint_detector.detect(frame)
        
        frame = self.drawer.draw_all(
            frame=frame, 
            box_detections=face_detections,
            keypoint_detections=keypoint_detections,
            label_text="Face",
        )
        
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
