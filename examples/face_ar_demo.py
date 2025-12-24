from utils import SourceFactory, SourceType, WindowDisplay
from pose_hand_detectors import KeypointDetectorFactory, KeypointDetectorType
from ar_effects import GlassesEffect, FrameEffect, HatEffect, FullFaceMaskEffect, BaseAREffect
from typing import List
import cv2
import logging


class FaceARApp:
    def __init__(self, effects: List[BaseAREffect], window_name="Face AR Demo", is_flip=True, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        
        self.face_mesh_detector = KeypointDetectorFactory.create(
            detector_type=KeypointDetectorType.MEDIAPIPE_FACE_MESH,
            logger=logger,
            max_num_faces=1,
            refine_landmarks=True,
        )
        
        self.effects = effects
        
        self.is_flip = is_flip
        self.logger = logger

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        
        face_detections = self.face_mesh_detector.detect(frame)
        
        for effect in self.effects:
            frame = effect.apply(frame, face_detections)
        
        return frame

    def run(self):
        if self.logger:
            self.logger.info("Starting Face AR application...")
        
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
    _logger = logging.getLogger('FaceAR')

    _effects = [
        FrameEffect(image_path="../assets/images/new_year_frame.png", logger=_logger),
        # HatEffect(image_path="assets/images/santa_hat.png", scale_factor=2.2, x_offset=0.0, y_offset=-0.5,
        #           logger=_logger),
        FullFaceMaskEffect(image_path="../assets/images/santa.png", scale_factor=1.7, x_offset=0.03, y_offset=0.13,
                           logger=_logger),
        GlassesEffect(image_path="../assets/images/glasses.png", scale_factor=1.8, logger=_logger),
    ]
    
    app = FaceARApp(effects=_effects, is_flip=True, logger=_logger)
    app.run()
