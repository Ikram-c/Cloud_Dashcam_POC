"""Quality-gated, illumination-aware, coverage-chunked frame extraction."""

from .config import Settings
from .extractor import FrameExtractor
from .pipeline import ExtractionPipeline
from .quality import FrameQualityGate
from .video_scanner import VideoScanner

__all__ = [
    "ExtractionPipeline",
    "FrameExtractor",
    "FrameQualityGate",
    "Settings",
    "VideoScanner",
]