"""Container-metadata probing for start time and GPS position.

Reconstructed module: the repository's copy of this file was
overwritten by the intrinsic (Weiss) module in a bad commit. The
public surface is defined by its call site
(``extractor.resolve_start_datetime``: ``dt, _ = probe(path)``) and
the README: ``pymediainfo`` reads the container's ``encoded_date``
(and GPS xyz tag when present); without the library or the native
libmediainfo, probing degrades to (None, None) and timestamp
resolution falls back to filename parsing.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pymediainfo import MediaInfo
    MEDIAINFO_AVAILABLE = True
except (ImportError, OSError):  # OSError: native libmediainfo missing
    MEDIAINFO_AVAILABLE = False

_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)
_XYZ_PATTERN = re.compile(
    r"(?P<lat>[+-]\d+(?:\.\d+)?)(?P<lon>[+-]\d+(?:\.\d+)?)"
)
_warned_unavailable = False


def _parse_encoded_date(raw: str) -> Optional[datetime]:
    """Parse a MediaInfo date string into an aware UTC datetime.

    MediaInfo emits variants like ``UTC 2024-03-15 08:30:00``,
    ``2024-03-15 08:30:00 UTC``, or a bare local-looking timestamp;
    bare timestamps are treated as UTC (containers rarely say).

    Args:
        raw (str): The raw tag value.

    Returns:
        Optional[datetime]: Aware UTC datetime, or None on failure.
    """
    text = raw.strip()
    utc_tagged = "UTC" in text.upper()
    text = re.sub(r"\bUTC\b", "", text, flags=re.IGNORECASE).strip()
    text = text.rstrip("Zz").strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    if not utc_tagged:
        logger.debug("Unparseable container date: %r", raw)
    return None


def _parse_xyz(raw: str) -> Optional[Tuple[float, float]]:
    """Parse an ISO 6709 xyz tag like ``+52.4500+001.7300/``.

    Args:
        raw (str): The raw tag value.

    Returns:
        Optional[Tuple[float, float]]: (lat, lon), or None.
    """
    match = _XYZ_PATTERN.search(raw.strip())
    if match is None:
        return None
    try:
        return float(match.group("lat")), float(match.group("lon"))
    except ValueError:
        return None


def probe(
    video_path: Path,
) -> Tuple[Optional[datetime], Optional[Tuple[float, float]]]:
    """Read the recording start time and GPS position from a container.

    Args:
        video_path (Path): The video file.

    Returns:
        Tuple[Optional[datetime], Optional[Tuple[float, float]]]:
            (aware UTC start time or None, (lat, lon) or None). Both
            are None when pymediainfo/libmediainfo is unavailable, the
            file is unreadable, or the tags are absent - the caller
            then falls back to filename parsing.
    """
    global _warned_unavailable
    if not MEDIAINFO_AVAILABLE:
        if not _warned_unavailable:
            logger.info(
                "pymediainfo/libmediainfo unavailable; container metadata "
                "disabled, falling back to filename timestamps"
            )
            _warned_unavailable = True
        return None, None
    try:
        info = MediaInfo.parse(str(video_path))
    except (OSError, RuntimeError, ValueError) as e:
        logger.debug("MediaInfo failed for %s: %s", video_path.name, e)
        return None, None
    dt: Optional[datetime] = None
    gps: Optional[Tuple[float, float]] = None
    for track in info.tracks:
        if track.track_type != "General":
            continue
        for attr in ("encoded_date", "tagged_date", "recorded_date"):
            raw = getattr(track, attr, None)
            if raw and dt is None:
                dt = _parse_encoded_date(str(raw))
        raw_xyz = getattr(track, "xyz", None)
        if raw_xyz and gps is None:
            gps = _parse_xyz(str(raw_xyz))
        break
    return dt, gps
