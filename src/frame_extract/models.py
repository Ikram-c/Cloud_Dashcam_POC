"""Immutable data models (NASA Power of 10, rules 3 and 6)."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .exceptions import SamplingExceedsDurationError


@dataclass(frozen=True, slots=True)
class FrameSlice:
    """A half-open range of frame indices [start, end)."""

    start: int
    end: int

    def __post_init__(self):
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    def __len__(self) -> int:
        return self.end - self.start

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.end))

    def __contains__(self, frame: int) -> bool:
        return self.start <= frame < self.end


@dataclass(frozen=True, slots=True)
class CropConfig:
    """A crop window; excludes dashcam OSD burn-in before all gates."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"Top-left must be non-negative, got ({self.x0},{self.y0})")
        if self.x1 <= self.x0:
            raise ValueError(f"x1 ({self.x1}) must exceed x0 ({self.x0})")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 ({self.y1}) must exceed y0 ({self.y0})")

    @property
    def width(self) -> int:
        """Crop width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """Crop height in pixels."""
        return self.y1 - self.y0

    def validate_against_frame(self, frame_h: int, frame_w: int):
        """Fail fast if the crop exceeds the frame.

        Args:
            frame_h (int): Frame height.
            frame_w (int): Frame width.

        Raises:
            ValueError: If the crop exceeds the frame bounds.
        """
        if self.x1 > frame_w or self.y1 > frame_h:
            raise ValueError(
                f"Crop ({self.x0},{self.y0})->({self.x1},{self.y1}) "
                f"exceeds frame {frame_w}x{frame_h}"
            )


@dataclass(frozen=True, slots=True)
class SampleConfig:
    """A time window within a video."""

    duration_seconds: float
    offset_seconds: float = 0.0

    def __post_init__(self):
        if self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive")
        if self.offset_seconds < 0.0:
            raise ValueError("offset_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic properties read from a video container."""

    total_frames: int
    fps: float
    frame_width: int = 0
    frame_height: int = 0

    def __post_init__(self):
        if self.total_frames <= 0:
            raise ValueError(f"total_frames must be positive, got {self.total_frames}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        return self.total_frames / self.fps

    def to_frame_slice(self, sample: Optional[SampleConfig]) -> FrameSlice:
        """Convert an optional sample window to a frame slice.

        Args:
            sample (Optional[SampleConfig]): Time window, or None for
                the full video.

        Returns:
            FrameSlice: The corresponding frame range.

        Raises:
            SamplingExceedsDurationError: If the window overruns.
        """
        if sample is None:
            return FrameSlice(0, self.total_frames)
        end_s = sample.offset_seconds + sample.duration_seconds
        if end_s > self.duration_seconds:
            raise SamplingExceedsDurationError(end_s, self.duration_seconds)
        start = min(int(round(sample.offset_seconds * self.fps)), self.total_frames)
        end = min(int(round(end_s * self.fps)), self.total_frames)
        return FrameSlice(start, end)


@dataclass(frozen=True, slots=True)
class QualityResult:
    """Outcome of the quality gates for one candidate frame.

    ``diff_from_previous`` is measured against the previous *evaluated*
    frame, whether or not that frame was kept.
    """

    accepted: bool
    reject_reason: Optional[str]
    blur_score: float
    mean_intensity: float
    diff_from_previous: Optional[float]


@dataclass(frozen=True, slots=True)
class GnssPoint:
    """One position fix."""

    timestamp: datetime
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class IlluminationPrior:
    """Forecast- and geometry-derived expectation for one timestamp."""

    timestamp: datetime
    sun_elevation_deg: float
    sun_azimuth_deg: float
    cloud_fraction: Optional[float]
    shortwave_wm2: Optional[float]
    expected_log_intensity: float
    daylight: bool


@dataclass(frozen=True, slots=True)
class IlluminationState:
    """Fused per-frame illumination estimate."""

    log_gain: float
    observed_log_mean: float
    expected_log_intensity: float
    confidence: float
    source: str


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    """One frame written to disk, with provenance and metrics.

    ``timestamp_sec`` derives from container PTS where the container's
    timestamps are sane (``timestamp_source == 'pts'``), else from
    frame_index / fps (``timestamp_source == 'index_fps'``). Within one
    video the source is always uniform: if PTS trust is lost mid-video,
    earlier frames are retroactively re-derived from index/fps.
    """

    video_path: Path
    frame_index: int
    timestamp_sec: float
    output_path: Path
    blur_score: float
    mean_intensity: float
    timestamp_source: str = "pts"
    sun_elevation_deg: Optional[float] = None
    cloud_fraction: Optional[float] = None
    log_gain: Optional[float] = None
    illum_confidence: Optional[float] = None
    normalized: bool = False


@dataclass(frozen=True, slots=True)
class VideoSummary:
    """Per-video extraction outcome."""

    video_path: Path
    vehicle: Optional[str]
    camera_id: Optional[str]
    file_datetime: Optional[datetime]
    fps: float
    total_frames: int
    frames_sampled: int
    frames_kept: int
    frames_rejected_blur: int
    frames_rejected_exposure: int
    frames_rejected_duplicate: int
    reflectance_frames: int


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One progress observation emitted during a pipeline run.

    ``stats`` is the extractor's live counter dict; treat it as
    read-only. In parallel mode events arrive only at per-video
    completion (callbacks cannot cross process boundaries), so
    ``frames_done == frames_total`` always holds there.
    """

    video_path: Path
    video_index: int
    video_count: int
    frames_done: int
    frames_total: int
    stats: dict
