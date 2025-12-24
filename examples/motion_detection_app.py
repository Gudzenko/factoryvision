import cv2
import numpy as np
from utils import SourceFactory, SourceType, WindowDisplay
from motion_detection.frame_difference_motion_detector import FrameDifferenceMotionDetector
import logging


class MotionDetectionApp:
    def __init__(self, window_name="Motion Detection", is_flip=False, threshold=30, min_area=500, logger=None):
        self.source = SourceFactory.create(
            source_type=SourceType.CAMERA,
            logger=logger,
        )
        self.window = WindowDisplay(window_name)
        self.detector = FrameDifferenceMotionDetector(threshold=threshold, min_area=min_area)
        self.is_flip = is_flip
        self.mode = 0

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
        result = self.detector.detect(frame)
        if self.mode == 1:
            mask = result['motion_mask']
            if mask is not None:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                display = cv2.addWeighted(frame, 0.5, mask_bgr, 0.5, 0)
            else:
                display = frame
            for bbox in result['motion_bbox']:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for contour in result['motion_contours']:
                pts = np.array(contour, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            for mv in result.get('motion_vectors', []):
                center = mv['center']
                dx, dy = mv['vector']
                tip = (int(center[0] + dx * 10), int(center[1] + dy * 10))
                cv2.arrowedLine(display, center, tip, (0, 0, 255), 2, tipLength=0.3)
            return display
        else:
            return frame

    def run(self):
        while True:
            frame = self.source.read()
            if frame is None:
                break
            display = self.process_frame(frame)
            self.window.show_frame(display)
            key = self.window.wait_key()
            if key == ord('1'):
                self.mode = 1
            elif key == ord('0'):
                self.mode = 0
            if self.window.should_close(key):
                break
        self.source.release()
        self.window.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger('MotionDetection')
    app = MotionDetectionApp(logger=_logger, is_flip=True)
    app.run()
