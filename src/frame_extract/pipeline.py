"""Directory-level orchestration: scan, extract, chunk, manifest, bridge."""

import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import pandas as pd
import yaml

from .config import Settings
from .coverage import CoverageChunker
from .exceptions import VideoOpenError
from .extractor import FrameExtractor
from .models import ExtractedFrame, ProgressEvent, VideoSummary
from .telemetry import TelemetryProvider
from .video_scanner import VideoScanner

logger = logging.getLogger(__name__)

THUMB_DIR_NAME = "_thumbs"


def _extract_one(
    args: Tuple[Settings, Path, Path, Optional[list]],
) -> Tuple[Path, tuple]:
    """Module-level worker for the process pool (fork-safe).

    Args:
        args (Tuple): (settings, video_path, output_dir, track_nodes).

    Returns:
        Tuple[Path, tuple]: (video_path, (frames, stats)).
    """
    settings, video_path, out_dir, nodes = args
    extractor = FrameExtractor(settings, track_nodes=nodes)
    return video_path, extractor.extract(video_path, out_dir)


class ExtractionPipeline:
    """Runs extraction over every video and writes the manifests."""

    def __init__(
        self, settings: Settings, telemetry: Optional[TelemetryProvider] = None,
    ):
        """Wire the scanner and the shared telemetry provider.

        Args:
            settings (Settings): Root configuration object.
            telemetry (Optional[TelemetryProvider]): Injected provider
                (for offline tests); created from settings when None.
        """
        self.settings = settings
        self.scanner = VideoScanner(settings.runtime)
        self.telemetry = telemetry or TelemetryProvider(settings)

    def run(
        self, progress: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> List[VideoSummary]:
        """Process every video, tolerating per-video failures.

        Telemetry resolves once up front. If it fails or is absent,
        illumination degrades to the fixed-location fallback and
        coverage chunking is skipped: chunking against a guessed
        position would be silently wrong rather than approximately
        right.

        Args:
            progress (Optional[Callable]): Receives ProgressEvent
                observations. In sequential mode (num_workers == 0)
                events arrive per sampled frame; in parallel mode,
                only at per-video completion.

        Returns:
            List[VideoSummary]: One summary per video completed.
        """
        videos = self.scanner.scan()
        output_root = Path(self.settings.runtime.output_directory)
        output_root.mkdir(parents=True, exist_ok=True)
        thumbs = output_root / THUMB_DIR_NAME
        if thumbs.exists():
            shutil.rmtree(thumbs)

        nodes = None
        needs_track = (
            self.settings.illumination.enabled
            or self.settings.coverage.enabled
        )
        if needs_track and self.telemetry.configured:
            try:
                nodes = self.telemetry.nodes(output_root)
            except (ImportError, FileNotFoundError, ValueError, OSError) as e:
                logger.warning(
                    "Telemetry unavailable (%s); illumination falls back to "
                    "the fixed location, coverage chunking is skipped", e,
                )
        elif needs_track:
            logger.info("No telemetry configured; using fixed-location fallback")

        summaries: List[VideoSummary] = []
        all_frames: List[ExtractedFrame] = []
        jobs = [(self.settings, p, output_root / p.stem, nodes) for p in videos]
        workers = self.settings.runtime.num_workers
        if workers > 0:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_extract_one, j) for j in jobs]
                completed = 0
                for future in as_completed(futures):
                    try:
                        video_path, (frames, stats) = future.result()
                    except (IOError, ValueError, VideoOpenError, cv2.error) as e:
                        logger.error("Extraction failed: %s", e)
                        completed += 1
                        continue
                    all_frames.extend(frames)
                    summaries.append(self._summarise(video_path, frames, stats))
                    if progress is not None:
                        progress(ProgressEvent(
                            video_path, completed, len(jobs), 1, 1, stats,
                        ))
                    completed += 1
        else:
            extractor = FrameExtractor(self.settings, track_nodes=nodes)
            for index, (_, video_path, out_dir, _) in enumerate(jobs):
                per_frame = None
                if progress is not None:
                    def per_frame(done, total, stats,
                                  _vp=video_path, _i=index):
                        progress(ProgressEvent(
                            _vp, _i, len(jobs), done, total, stats,
                        ))
                try:
                    frames, stats = extractor.extract(
                        video_path, out_dir, progress=per_frame,
                    )
                except (IOError, ValueError, VideoOpenError, cv2.error) as e:
                    logger.error("Extraction failed for %s: %s", video_path.name, e)
                    continue
                all_frames.extend(frames)
                summaries.append(self._summarise(video_path, frames, stats))
                if progress is not None:
                    progress(ProgressEvent(
                        video_path, index, len(jobs), 1, 1, stats,
                    ))
        self._write_manifest(all_frames, summaries, output_root)
        if self.settings.output.export_kept_slices:
            self._export_kept_slices(all_frames, output_root)
        if self.settings.coverage.enabled and nodes is not None:
            try:
                CoverageChunker(self.settings).run(all_frames, output_root, nodes)
            except (ImportError, ValueError, OSError) as e:
                logger.error("Coverage chunking failed: %s", e)
        return summaries

    def _summarise(self, video_path: Path, frames, stats) -> VideoSummary:
        """Build the per-video summary record.

        Args:
            video_path (Path): The video.
            frames: Kept frames.
            stats: Extraction statistics dict.

        Returns:
            VideoSummary: The summary.
        """
        meta = (
            self.scanner.parse_filename(video_path)
            if self.settings.runtime.parse_filenames else None
        ) or {}
        return VideoSummary(
            video_path=video_path,
            vehicle=meta.get("vehicle"),
            camera_id=meta.get("camera_id"),
            file_datetime=meta.get("file_datetime"),
            fps=stats["fps"],
            total_frames=stats["total_frames"],
            frames_sampled=stats["sampled"],
            frames_kept=len(frames),
            frames_rejected_blur=stats["blur"],
            frames_rejected_exposure=stats["underexposed"] + stats["overexposed"],
            frames_rejected_duplicate=stats["duplicate"],
            reflectance_frames=stats["reflectance"],
        )

    def _write_manifest(self, frames, summaries, output_root: Path):
        """Write the frame manifest and per-video summary CSVs.

        Args:
            frames: All kept frames across videos.
            summaries: All video summaries.
            output_root (Path): Output directory.
        """
        if frames:
            df = pd.DataFrame([{
                "video_path": str(f.video_path),
                "frame_index": f.frame_index,
                "timestamp_sec": round(f.timestamp_sec, 3),
                "timestamp_source": f.timestamp_source,
                "output_path": str(f.output_path),
                "blur_score": round(f.blur_score, 1),
                "mean_intensity": round(f.mean_intensity, 1),
                "sun_elevation_deg": (
                    round(f.sun_elevation_deg, 2)
                    if f.sun_elevation_deg is not None else None
                ),
                "cloud_fraction": f.cloud_fraction,
                "log_gain": (
                    round(f.log_gain, 4) if f.log_gain is not None else None
                ),
                "illum_confidence": f.illum_confidence,
                "normalized": f.normalized,
            } for f in frames])
            path = output_root / self.settings.output.manifest_name
            df.to_csv(path, index=False)
            logger.info("Manifest: %s (%d frames)", path, len(df))
        if summaries:
            pd.DataFrame([asdict(s) for s in summaries]).to_csv(
                output_root / "video_summaries.csv", index=False
            )

    def _export_kept_slices(self, frames: List[ExtractedFrame], output_root: Path):
        """Write per-video kept_slices.yaml for the video-zarr bridge.

        Args:
            frames (List[ExtractedFrame]): All kept frames.
            output_root (Path): Output directory.
        """
        by_video: dict = {}
        for f in frames:
            by_video.setdefault(f.video_path, []).append(f)
        for video_path, video_frames in by_video.items():
            every = self._effective_every(video_frames)
            ranges = self._kept_slices(video_frames, every)
            payload = {
                "source_video": str(video_path),
                "every": every,
                "slices": ranges,
            }
            dest = output_root / video_path.stem / "kept_slices.yaml"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(yaml.safe_dump(payload, sort_keys=False))
            logger.info("Kept slices: %s (%d ranges)", dest, len(ranges))

    @staticmethod
    def _effective_every(frames: List[ExtractedFrame]) -> int:
        """Infer the sampling stride from kept frame indices.

        Args:
            frames (List[ExtractedFrame]): Kept frames, one video.

        Returns:
            int: The smallest positive index gap, at least 1.
        """
        indices = sorted(f.frame_index for f in frames)
        gaps = [b - a for a, b in zip(indices, indices[1:]) if b > a]
        return min(gaps) if gaps else 1

    @staticmethod
    def _kept_slices(frames: List[ExtractedFrame], every: int) -> list:
        """Collapse kept frame indices into contiguous [start, end) ranges.

        Args:
            frames (List[ExtractedFrame]): Kept frames, one video.
            every (int): Sampling stride used during extraction.

        Returns:
            list: Dicts of {start, end} in source-frame indices,
                directly consumable as video_zarr FrameSlice ranges.
        """
        indices = sorted(f.frame_index for f in frames)
        if not indices:
            return []
        ranges, run_start, prev = [], indices[0], indices[0]
        for idx in indices[1:]:
            if idx - prev > every:
                ranges.append({"start": run_start, "end": prev + 1})
                run_start = idx
            prev = idx
        ranges.append({"start": run_start, "end": prev + 1})
        return ranges