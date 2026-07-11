"""Video capture context manager and fault-tolerant frame iterator.

Adapted from the video_zarr package: RAII capture handling, tolerance
of corrupt mid-file segments up to a bounded failure budget, and
container PTS timestamps attached to every yielded frame.
"""

import logging
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from .exceptions import VideoOpenError
from .models import FrameSlice, VideoMetadata

logger = logging.getLogger(__name__)


class VideoCapture:
    """Context-managed cv2.VideoCapture."""

    __slots__ = ("_path", "_capture")

    def __init__(self, path: Path):
        """Store the path; the capture opens on __enter__.

        Args:
            path (Path): Video file path.
        """
        self._path = path
        self._capture = None

    def __enter__(self) -> cv2.VideoCapture:
        """Open the capture.

        Returns:
            cv2.VideoCapture: The opened capture.

        Raises:
            VideoOpenError: If the file cannot be opened.
        """
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise VideoOpenError(self._path)
        return self._capture

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Release the capture."""
        if self._capture is not None:
            self._capture.release()
        return False


def read_metadata(video_path: Path) -> VideoMetadata:
    """Read frame count, fps, and dimensions from a video.

    Args:
        video_path (Path): Video file path.

    Returns:
        VideoMetadata: Validated metadata.

    Raises:
        VideoOpenError: If the video cannot be opened or reports
            non-positive frame count or fps.
    """
    with VideoCapture(video_path) as cap:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or fps <= 0:
        raise VideoOpenError(video_path)
    return VideoMetadata(total_frames=total, fps=fps, frame_width=w, frame_height=h)


class TimestampedFrameIterator:
    """Yields (index, pts_ms, frame), tolerating corrupt segments.

    Read failures are skipped up to ``max_fails`` *consecutive*
    failures, after which iteration stops; any successful read resets
    the budget. Both the frame range and the failure budget are hard
    bounds (Po10 rule 2).

    Known limitation: a failed ``read()`` may or may not have consumed
    a container frame, so on corrupt files the reported index (and any
    index/fps-derived timestamp) can drift relative to true source
    indices. Downstream consumers of kept-slice ranges should treat
    ranges from corrupt files as approximate.
    """

    __slots__ = ("_cap", "_slice", "_every", "_max_fails", "_pos", "_fails")

    def __init__(
        self,
        capture: cv2.VideoCapture,
        frame_slice: FrameSlice,
        every: int,
        max_fails: int,
    ):
        """Position the capture and set bounds.

        Args:
            capture (cv2.VideoCapture): An opened capture.
            frame_slice (FrameSlice): Frame index range to read.
            every (int): Yield every N-th frame.
            max_fails (int): Consecutive read failures tolerated.

        Raises:
            ValueError: If every or max_fails is not positive.
        """
        if every <= 0:
            raise ValueError("every must be positive")
        if max_fails <= 0:
            raise ValueError("max_fails must be positive")
        self._cap = capture
        self._slice = frame_slice
        self._every = every
        self._max_fails = max_fails
        self._pos = frame_slice.start
        self._fails = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_slice.start)

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[int, float, np.ndarray]:
        while self._pos < self._slice.end:
            if self._fails > self._max_fails:
                logger.warning("Failure budget exhausted at frame %d", self._pos)
                raise StopIteration
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._fails += 1
                self._pos += 1
                continue
            self._fails = 0  # budget counts *consecutive* failures only
            pos = self._pos
            pts_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            self._pos += 1
            if (pos - self._slice.start) % self._every == 0:
                return pos, pts_ms, frame
        raise StopIteration