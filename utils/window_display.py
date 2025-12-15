import cv2


class WindowDisplay:
    def __init__(self, window_name="FactoryVision"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay=1):
        return cv2.waitKey(delay) & 0xFF

    def should_close(self, key):
        return key == 27

    def close(self):
        cv2.destroyWindow(self.window_name)
