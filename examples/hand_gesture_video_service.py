from utils import SourceFactory, SourceType, WindowDisplay
from hand_gesture_recognition.hand_gesture_recognizer import HandGestureRecognizer
import logging
import cv2


class HandGestureVideoService:
    def __init__(self, window_name="Hand Gesture Recognition", is_flip=False, device='cpu', confidence_threshold=0.5, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        self.recognizer = HandGestureRecognizer(
            device=device, 
            confidence_threshold=confidence_threshold,
        )
        self.is_flip = is_flip

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        gesture, probs = self.recognizer.predict(frame)
        if gesture:
            prob = probs.get(gesture, 0) if probs else 0
            cv2.putText(frame, f"Gesture: {gesture} ({prob:.2f})", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    _logger = logging.getLogger('HandGestureVideoService')
    app = HandGestureVideoService(
        logger=_logger, 
        is_flip=True,
        confidence_threshold=0.7,
    )
    app.run()
