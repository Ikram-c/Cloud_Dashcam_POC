"""Bounded video-directory scanning and filename metadata parsing.

Reconstructed module: the original was lost from the repository. The
public surface is defined by its call sites (extractor, pipeline,
coverage) and the README's filename convention
``VEHICLE_YYYYMMDD_HHMMSS_CAMERA.ext``.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import RuntimeConfig

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(
    r"^(?P<vehicle>[A-Za-z0-9-]+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})_"
    r"(?P<camera>[A-Za-z0-9-]+)$"
)


class VideoScanner:
    """Finds video files within hard bounds and parses their names."""

    def __init__(self, config: RuntimeConfig):
        """Initialise the scanner.

        Args:
            config (RuntimeConfig): Validated runtime configuration.
        """
        self.config = config

    def scan(self) -> List[Path]:
        """List video files in the configured directory, bounded.

        Non-recursive by design: one directory per survey run. The
        result is sorted for deterministic processing order and capped
        at ``max_video_files`` (Po10 rule 2), with a warning when the
        cap truncates.

        Returns:
            List[Path]: Sorted video paths; empty when the directory
                does not exist.
        """
        root = Path(self.config.video_directory)
        if not root.is_dir():
            logger.warning("Video directory not found: %s", root)
            return []
        extensions = {e.lower() for e in self.config.video_extensions}
        videos = sorted(
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if len(videos) > self.config.max_video_files:
            logger.warning(
                "Found %d videos; processing the first %d (max_video_files)",
                len(videos), self.config.max_video_files,
            )
            videos = videos[:self.config.max_video_files]
        logger.info("Found %d video(s) in %s", len(videos), root)
        return videos

    def parse_filename(self, video_path: Path) -> Optional[dict]:
        """Parse vehicle, timestamp, and camera from a filename.

        Expected stem format: ``VEHICLE_YYYYMMDD_HHMMSS_CAMERA``.
        The parsed datetime is naive; the caller attaches the
        configured timezone (extractor.resolve_start_datetime).

        Args:
            video_path (Path): The video file.

        Returns:
            Optional[dict]: {'vehicle', 'camera_id', 'file_datetime'},
                or None when the stem does not match.
        """
        match = FILENAME_PATTERN.match(video_path.stem)
        if match is None:
            logger.debug("Unparseable filename: %s", video_path.name)
            return None
        try:
            file_dt = datetime.strptime(
                match.group("date") + match.group("time"), "%Y%m%d%H%M%S",
            )
        except ValueError:
            logger.debug("Invalid timestamp in filename: %s", video_path.name)
            return None
        return {
            "vehicle": match.group("vehicle"),
            "camera_id": match.group("camera"),
            "file_datetime": file_dt,
        }
