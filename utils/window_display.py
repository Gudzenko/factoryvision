import cv2
import os
from datetime import datetime


class WindowDisplay:
    def __init__(self, window_name="FactoryVision", screenshot_dir="../assets/images/screens", screenshot_key=ord('s')):
        self.window_name = window_name
        self.screenshot_dir = screenshot_dir
        self.screenshot_key = screenshot_key
        self._last_frame = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def show_frame(self, frame):
        self._last_frame = frame.copy()
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay=1):
        key = cv2.waitKey(delay) & 0xFF
        if key == self.screenshot_key:
            self.save_screenshot()
        return key

    def save_screenshot(self):
        if hasattr(self, '_last_frame') and self._last_frame is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{now}.png"
            path = os.path.join(self.screenshot_dir, filename)
            cv2.imwrite(path, self._last_frame)

    def should_close(self, key):
        return key == 27

    def close(self):
        cv2.destroyWindow(self.window_name)
