import cv2
import numpy as np
import logging
import time
from typing import Optional
from .base_source import BaseSource


class VideoFileStream(BaseSource):
    def __init__(self, video_path: str, loop: bool = True, realtime: bool = False, 
                 speed_factor: float = 1.0, logger: Optional[logging.Logger] = None):
        self.video_path = video_path
        self.loop = loop
        self.realtime = realtime
        self.speed_factor = speed_factor
        self.logger = logger
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_delay = (1.0 / self.fps * self.speed_factor) if self.fps > 0 else 0.033
        self.last_frame_time = None
        
        if self.logger:
            mode = "realtime" if self.realtime else "fast"
            self.logger.info(f"Video loaded: {video_path} ({self.total_frames} frames, "
                             f"{self.fps:.2f} FPS, mode={mode}, speed={speed_factor}x)")

    def read(self) -> np.ndarray:
        if self.realtime and self.last_frame_time is not None:
            elapsed = time.time() - self.last_frame_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if self.realtime:
            self.last_frame_time = time.time()
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                return None
        
        return frame if ret else None

    def release(self):
        self.cap.release()
        if self.logger:
            self.logger.info(f"Video released: {self.video_path}")

    def release(self):
        self.cap.release()
        if self.logger:
            self.logger.info(f"Video released: {self.video_path}")
