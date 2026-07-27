"""Package exceptions."""

from pathlib import Path


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened."""

    __slots__ = ("path",)

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Failed to open video: {path}")


class SamplingExceedsDurationError(Exception):
    """Raised when a sample window extends past the video end."""

    __slots__ = ("requested", "available")

    def __init__(self, requested: float, available: float):
        self.requested = requested
        self.available = available
        super().__init__(
            f"Requested {requested:.2f}s exceeds video duration of {available:.2f}s"
        )
