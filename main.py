from utils.camera_stream import CameraStream
from utils.window_display import WindowDisplay
import cv2


class FactoryVisionApp:
    def __init__(self, window_name="FactoryVision Demo", is_flip=False):
        self.camera = CameraStream()
        self.window = WindowDisplay(window_name)
        self.is_flip = is_flip

    def process_frame(self, frame):
        if self.is_flip:
            frame = cv2.flip(frame, 1)
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
    app = FactoryVisionApp(is_flip=True)
    app.run()
