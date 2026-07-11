"""Bounded, quality-gated, illumination-aware extraction of a video."""

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone as tz
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from . import media_metadata
from .capture import TimestampedFrameIterator, VideoCapture, read_metadata
from .config import Settings
from .illumination import IlluminationEstimator
from .intrinsic import WeissReflectanceEstimator
from .models import ExtractedFrame, FrameSlice, QualityResult
from .quality import FrameQualityGate
from .video_scanner import VideoScanner

logger = logging.getLogger(__name__)


def resolve_start_datetime(
    settings: Settings, scanner: VideoScanner, video_path: Path,
) -> Optional[datetime]:
    """Resolve a video's start time: container metadata, then filename.

    Args:
        settings (Settings): Root configuration.
        scanner (VideoScanner): Filename parser.
        video_path (Path): The video file.

    Returns:
        Optional[datetime]: Timezone-aware start time, or None.
    """
    source = settings.input.metadata_source
    if source == "none":
        return None
    if source in ("auto", "container"):
        dt, _ = media_metadata.probe(video_path)
        if dt is not None:
            return dt
        if source == "container":
            return None
    if settings.runtime.parse_filenames:
        meta = scanner.parse_filename(video_path)
        if meta is not None:
            zone = ZoneInfo(settings.illumination.timezone)
            return meta["file_datetime"].replace(tzinfo=zone).astimezone(tz.utc)
    return None


class FrameExtractor:
    """Reads a video within hard bounds and writes accepted frames."""

    def __init__(self, settings: Settings, track_nodes: Optional[list] = None):
        """Wire gates and estimators from settings.

        Args:
            settings (Settings): Root configuration object.
            track_nodes (Optional[list]): Shared GPX telemetry nodes,
                passed through to the illumination estimator.
        """
        self.settings = settings
        self.gate = FrameQualityGate(settings.quality)
        self.scanner = VideoScanner(settings.runtime)
        self.intrinsic = (
            WeissReflectanceEstimator(settings.intrinsic)
            if settings.intrinsic.enabled else None
        )
        self.illumination = (
            IlluminationEstimator(settings.illumination, track_nodes)
            if settings.illumination.enabled else None
        )
        self.crop = settings.input.crop
        self.sample = settings.input.sample
        self._segment_counter = 0

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        progress: Optional[Callable[[int, int, dict], None]] = None,
    ) -> Tuple[List[ExtractedFrame], dict]:
        """Extract quality-gated frames from one video.

        Args:
            video_path (Path): Path to the video file.
            output_dir (Path): Directory for this video's frames.
            progress (Optional[Callable]): Called after each sampled
                frame with (frames_done, frames_total, stats), where
                frames_done/frames_total are positions within this
                video's slice. Must be cheap and must not raise.

        Returns:
            Tuple[List[ExtractedFrame], dict]: Kept frames and a stats
                dict with fps, total_frames (whole video), slice_frames
                (frames in the processed range), sampled, per-reason
                rejection counts, and reflectance frame count.

        Raises:
            VideoOpenError: If the video cannot be opened.
            ValueError: If a configured crop exceeds the frame.
        """
        metadata = read_metadata(video_path)
        if self.crop is not None:
            self.crop.validate_against_frame(metadata.frame_height, metadata.frame_width)
        full_slice = metadata.to_frame_slice(self.sample)
        capped_end = min(
            full_slice.end,
            full_slice.start + self.settings.runtime.max_video_frames,
        )
        frame_slice = FrameSlice(full_slice.start, capped_end)
        fps = metadata.fps or self.settings.runtime.default_fps
        interval = self._sample_interval(fps)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_dt = self._resolve_start_datetime(video_path)
        illum_active = self.illumination is not None and base_dt is not None
        if self.illumination is not None and base_dt is None:
            logger.warning(
                "%s: no parseable timestamp; illumination disabled for this video",
                video_path.name,
            )
        if illum_active:
            self.illumination.reset()
        self.gate.reset()
        if self.intrinsic is not None:
            self.intrinsic.reset()
        self._segment_counter = 0

        kept: List[ExtractedFrame] = []
        stats = {
            "fps": fps,
            "total_frames": metadata.total_frames,
            "slice_frames": len(frame_slice),
            "sampled": 0,
            "blur": 0, "underexposed": 0, "overexposed": 0, "duplicate": 0,
            "reflectance": 0,
        }
        pts_trusted = True
        prev_pts = -1.0
        logger.info(
            "Extracting %s (%d frames @ %.1f fps, interval %d)",
            video_path.name, len(frame_slice), fps, interval,
        )
        with VideoCapture(video_path) as cap:
            iterator = TimestampedFrameIterator(
                cap, frame_slice, every=interval,
                max_fails=self.settings.runtime.max_consecutive_fails,
            )
            for frame_idx, pts_ms, frame in iterator:
                if pts_trusted and (pts_ms <= prev_pts or (frame_idx > 0 and pts_ms == 0.0)):
                    logger.warning(
                        "%s: untrusted container PTS; using index/fps",
                        video_path.name,
                    )
                    pts_trusted = False
                    kept = [
                        replace(
                            f,
                            timestamp_sec=f.frame_index / fps,
                            timestamp_source="index_fps",
                        )
                        for f in kept
                    ]
                prev_pts = pts_ms
                timestamp_sec = (
                    pts_ms / 1000.0 if pts_trusted else frame_idx / fps
                )
                if self.crop is not None:
                    frame = frame[self.crop.y0:self.crop.y1, self.crop.x0:self.crop.x1]
                stats["sampled"] += 1
                if progress is not None:
                    progress(
                        frame_idx - frame_slice.start + 1,
                        len(frame_slice),
                        stats,
                    )

                prior, state, band = None, None, None
                if illum_active:
                    dt = base_dt + timedelta(seconds=timestamp_sec)
                    outcome = self.illumination.update(dt, frame)
                    if outcome is not None:
                        prior, state = outcome
                        band = self.illumination.exposure_band(prior)

                result = self.gate.evaluate(frame, exposure_band=band)
                if not result.accepted:
                    stats[result.reject_reason] += 1
                    if result.reject_reason == "duplicate":
                        if self.intrinsic is not None:
                            self.intrinsic.add_frame(frame)
                    else:
                        stats["reflectance"] += self._flush_intrinsic(
                            video_path, output_dir
                        )
                    continue

                stats["reflectance"] += self._flush_intrinsic(video_path, output_dir)
                to_write = frame
                normalized = False
                if (illum_active and state is not None
                        and self.settings.illumination.normalize_output):
                    to_write = self.illumination.normalize(frame)
                    normalized = True
                saved = self._write_frame(to_write, video_path, output_dir,
                                          frame_idx, result)
                if saved is not None:
                    kept.append(ExtractedFrame(
                        video_path=video_path, frame_index=frame_idx,
                        timestamp_sec=timestamp_sec, output_path=saved,
                        blur_score=result.blur_score,
                        mean_intensity=result.mean_intensity,
                        timestamp_source="pts" if pts_trusted else "index_fps",
                        sun_elevation_deg=prior.sun_elevation_deg if prior else None,
                        cloud_fraction=prior.cloud_fraction if prior else None,
                        log_gain=state.log_gain if state else None,
                        illum_confidence=state.confidence if state else None,
                        normalized=normalized,
                    ))
                if len(kept) >= self.settings.sampling.max_frames_per_video:
                    logger.info("Reached max_frames_per_video for %s", video_path.name)
                    break
        stats["reflectance"] += self._flush_intrinsic(video_path, output_dir)
        logger.info(
            "%s: kept %d of %d sampled", video_path.name, len(kept), stats["sampled"]
        )
        return kept, stats

    def _sample_interval(self, fps: float) -> int:
        """Resolve the frame interval from the sampling mode.

        Args:
            fps (float): Video frame rate.

        Returns:
            int: Frames between samples, at least 1.
        """
        c = self.settings.sampling
        if c.mode == "frames":
            return c.every_n_frames
        return max(1, int(round(fps * c.every_n_seconds)))

    def _resolve_start_datetime(self, video_path: Path) -> Optional[datetime]:
        """Delegate to the module-level resolver."""
        return resolve_start_datetime(self.settings, self.scanner, video_path)

    def _flush_intrinsic(self, video_path: Path, output_dir: Path) -> int:
        """Emit a reflectance frame if the static-segment buffer is ready.

        Args:
            video_path (Path): Source video (for naming).
            output_dir (Path): Per-video output directory.

        Returns:
            int: 1 if a reflectance frame was written, else 0.
        """
        if self.intrinsic is None or self.intrinsic.frame_count == 0:
            return 0
        written = 0
        if self.intrinsic.ready:
            reflectance = self.intrinsic.estimate_reflectance()
            if reflectance is not None:
                ref_dir = output_dir / "reflectance"
                ref_dir.mkdir(parents=True, exist_ok=True)
                name = (
                    f"{video_path.stem}_seg{self._segment_counter:03d}"
                    f"_T{self.intrinsic.frame_count}.png"
                )
                if cv2.imwrite(str(ref_dir / name), reflectance):
                    written = 1
                    logger.info("Reflectance frame: %s", name)
                self._segment_counter += 1
        self.intrinsic.reset()
        return written

    def _write_frame(
        self,
        frame: np.ndarray,
        video_path: Path,
        output_dir: Path,
        frame_idx: int,
        result: QualityResult,
    ) -> Optional[Path]:
        """Optionally resize and write one accepted frame.

        Args:
            frame (np.ndarray): The frame to write.
            video_path (Path): Source video (for the filename stem).
            output_dir (Path): Destination directory.
            frame_idx (int): Frame index within the video.
            result (QualityResult): Metrics encoded into the filename.

        Returns:
            Optional[Path]: The written path, or None on failure.
        """
        c = self.settings.output
        out = frame
        if c.resize_width is not None and frame.shape[1] != c.resize_width:
            scale = c.resize_width / frame.shape[1]
            out = cv2.resize(
                frame, (c.resize_width, int(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        name = (
            f"{video_path.stem}_f{frame_idx:06d}"
            f"_blur{result.blur_score:.0f}.{c.image_format}"
        )
        path = output_dir / name
        params = (
            [cv2.IMWRITE_JPEG_QUALITY, c.jpeg_quality] if c.image_format == "jpg" else []
        )
        if not cv2.imwrite(str(path), out, params):
            logger.warning("Failed to write frame: %s", path)
            return None
        return path