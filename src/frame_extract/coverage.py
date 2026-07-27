"""Post-extraction chunking of kept frames by network coverage.

The GPX telemetry defines where the vehicle was; the selected
carrier's zones define where connectivity exists. Subdividing the
route at zone boundaries yields time intervals labelled by zone; each
kept frame's absolute timestamp is looked up against those intervals,
and frames are then collapsed into per-zone, per-video slice ranges
compatible with the video-zarr bridge format.
"""

import bisect
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import Settings
from .extractor import resolve_start_datetime
from .models import ExtractedFrame
from .route_graph import NO_COVERAGE, create_route_dag, subdivide_route_fast
from .video_scanner import VideoScanner

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CoverageInterval:
    """One labelled span of the route in time."""

    start: datetime
    end: datetime
    zone: str

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError("interval end must not precede start")


class CoverageChunker:
    """Assigns kept frames to coverage zones and writes chunk manifests."""

    def __init__(self, settings: Settings):
        """Wire the scanner.

        Args:
            settings (Settings): Root configuration.
        """
        self.settings = settings
        self.config = settings.coverage
        self.scanner = VideoScanner(settings.runtime)

    def run(
        self, frames: List[ExtractedFrame], output_root: Path, nodes: List[dict],
    ) -> dict:
        """Chunk kept frames by coverage and write the chunk manifest.

        Args:
            frames (List[ExtractedFrame]): All kept frames.
            output_root (Path): Pipeline output directory.
            nodes (List[dict]): Shared GPX telemetry nodes.

        Returns:
            dict: zone -> list of {video, start, end} slice ranges.
        """
        intervals = self.build_intervals(nodes)
        logger.info(
            "Coverage (%s): %d intervals across %d zones",
            self.config.network, len(intervals), len(self.config.zones),
        )
        assignments = self._assign(frames, intervals)
        chunks = self._collapse(assignments)
        payload = {"network": self.config.network, "zones": chunks}
        dest = output_root / self.config.chunk_manifest_name
        dest.write_text(yaml.safe_dump(payload, sort_keys=False))
        logger.info("Coverage chunks: %s", dest)
        return chunks

    def build_intervals(self, nodes: List[dict]) -> List["CoverageInterval"]:
        """Subdivide the route and emit sorted labelled time intervals.

        Edges whose endpoint times are unavailable are skipped: they
        cannot place frames in time. Where subdivided intervals
        overlap (overlapping zones), the earliest-starting interval
        wins at lookup, per zone_at.

        Args:
            nodes (List[dict]): GPX telemetry nodes.

        Returns:
            List[CoverageInterval]: Sorted by start time.
        """
        dag = create_route_dag(nodes)
        subdivided = subdivide_route_fast(dag, dict(self.config.zones))
        intervals: List[CoverageInterval] = []
        for u, v, data in subdivided.edges(data=True):
            t_u = subdivided.nodes[u]["time"]
            t_v = subdivided.nodes[v]["time"]
            if t_u is None or t_v is None:
                continue
            zone = data.get("coverage", NO_COVERAGE)
            intervals.append(CoverageInterval(t_u, t_v, zone))
        intervals.sort(key=lambda iv: iv.start)
        return intervals

    @staticmethod
    def zone_at(
        intervals: List["CoverageInterval"], dt: datetime,
    ) -> str:
        """Look up the zone label at a timestamp.

        Args:
            intervals (List[CoverageInterval]): Sorted intervals.
            dt (datetime): Timezone-aware query time.

        Returns:
            str: The containing interval's zone, or NO_COVERAGE when
                the timestamp is outside the track.

        Raises:
            ValueError: If the timestamp is naive.
        """
        if dt.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        starts = [iv.start for iv in intervals]
        idx = bisect.bisect_right(starts, dt) - 1
        while idx >= 0:
            if intervals[idx].end >= dt:
                return intervals[idx].zone
            idx -= 1
        return NO_COVERAGE

    def _assign(
        self, frames: List[ExtractedFrame], intervals: List["CoverageInterval"],
    ) -> Dict[str, Dict[Path, List[ExtractedFrame]]]:
        """Map every frame to a zone, grouped zone -> video -> frames.

        Videos without a resolvable start time are assigned wholesale
        to NO_COVERAGE, with a warning.

        Args:
            frames (List[ExtractedFrame]): All kept frames.
            intervals (List[CoverageInterval]): Sorted intervals.

        Returns:
            Dict: zone -> video path -> that video's frames in zone.
        """
        base_dt_cache: Dict[Path, Optional[datetime]] = {}
        grouped: Dict[str, Dict[Path, List[ExtractedFrame]]] = {}
        for frame in frames:
            video = frame.video_path
            if video not in base_dt_cache:
                base_dt_cache[video] = resolve_start_datetime(
                    self.settings, self.scanner, video,
                )
                if base_dt_cache[video] is None:
                    logger.warning(
                        "%s: no timestamp; frames assigned to %s",
                        video.name, NO_COVERAGE,
                    )
            base = base_dt_cache[video]
            if base is None:
                zone = NO_COVERAGE
            else:
                dt = base + timedelta(seconds=frame.timestamp_sec)
                zone = self.zone_at(intervals, dt)
            grouped.setdefault(zone, {}).setdefault(video, []).append(frame)
        return grouped

    @staticmethod
    def _collapse(
        grouped: Dict[str, Dict[Path, List[ExtractedFrame]]],
    ) -> dict:
        """Collapse per-zone frames into video-zarr-style slice ranges.

        Deferred pipeline import: pipeline.py imports this module at
        top level, so a module-level import back would be circular.

        Args:
            grouped (Dict): zone -> video -> frames.

        Returns:
            dict: zone -> [{video, start, end}, ...], end exclusive.
        """
        from .pipeline import ExtractionPipeline
        out: dict = {}
        for zone, by_video in grouped.items():
            entries = []
            for video, video_frames in by_video.items():
                every = ExtractionPipeline._effective_every(video_frames)
                for r in ExtractionPipeline._kept_slices(video_frames, every):
                    entries.append({
                        "video": str(video),
                        "start": r["start"],
                        "end": r["end"],
                    })
            out[zone] = sorted(entries, key=lambda e: (e["video"], e["start"]))
        return out
