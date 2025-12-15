import logging
from typing import Optional
from enum import Enum

from .camera_stream import CameraStream
from .video_file_stream import VideoFileStream
from .base_source import BaseSource


class SourceType(Enum):
    CAMERA = "camera"
    VIDEO_FILE = "video"


class SourceFactory:
    @staticmethod
    def create(source_type: SourceType, logger: Optional[logging.Logger] = None, **kwargs) -> BaseSource:
        if source_type == SourceType.CAMERA:
            return CameraStream(logger=logger, **kwargs)
        
        elif source_type == SourceType.VIDEO_FILE:
            return VideoFileStream(logger=logger, **kwargs)
        
        else:
            raise ValueError(f"Unknown source type: {source_type}")
