import cv2


class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera with ID {camera_id}")

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
