"""Bounded video-directory scanning and filename metadata parsing.

Filenames following ``VEHICLE_YYYYMMDD_HHMMSS_CAMERA.ext`` carry the
capture start time used by illumination and coverage when container
metadata is absent. Non-conforming names still extract; they simply
resolve no timestamp.
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
        """Store the runtime configuration.

        Args:
            config (RuntimeConfig): Paths, extensions, and bounds.
        """
        self.config = config

    def scan(self) -> List[Path]:
        """Return the sorted, bounded list of video files.

        Returns:
            List[Path]: Sorted paths whose suffix matches the
                configured extensions, capped at max_video_files.

        Raises:
            FileNotFoundError: If the video directory does not exist.
        """
        directory = Path(self.config.video_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Video directory not found: {directory}")
        extensions = {e.lower() for e in self.config.video_extensions}
        videos = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )
        if len(videos) > self.config.max_video_files:
            logger.warning(
                "Found %d videos; bounding to max_video_files=%d",
                len(videos), self.config.max_video_files,
            )
            videos = videos[: self.config.max_video_files]
        logger.info("Scanned %s: %d videos", directory, len(videos))
        return videos

    def parse_filename(self, video_path: Path) -> Optional[dict]:
        """Parse vehicle, timestamp, and camera from a filename.

        Args:
            video_path (Path): Any path; only the stem is inspected.

        Returns:
            Optional[dict]: {vehicle, camera_id, file_datetime} with a
                naive datetime (caller localises), or None when the
                stem does not match the expected pattern.
        """
        match = FILENAME_PATTERN.match(video_path.stem)
        if match is None:
            return None
        try:
            file_datetime = datetime.strptime(
                match.group("date") + match.group("time"), "%Y%m%d%H%M%S",
            )
        except ValueError:
            return None
        return {
            "vehicle": match.group("vehicle"),
            "camera_id": match.group("camera"),
            "file_datetime": file_datetime,
        }
