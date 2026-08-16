"""Optional container-metadata probe: capture datetime and GPS.

Reconstructed module (the previous file had been overwritten with a
copy of the Weiss estimator). Contract, per ``extractor.py`` and the
README: ``probe(path)`` returns ``(datetime | None, (lat, lon) | None)``.

pymediainfo — and the native libmediainfo it wraps — is an optional
dependency (the ``metadata`` extra). When it is missing, unparseable,
or the container carries no date, ``probe`` returns ``(None, None)``
and never raises; callers then fall back to filename parsing
(``input.metadata_source: auto``).
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Track fields checked for a capture date, in preference order.
_DATE_KEYS = ("encoded_date", "tagged_date", "recorded_date", "mastered_date")

# ISO 6709 location string, e.g. "+53.8050-001.5500/" (lat then lon).
_ISO6709 = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)\s*([+-]\d+(?:\.\d+)?)")


def _parse_date(raw: str) -> Optional[datetime]:
    """Parse a MediaInfo date string into an aware UTC datetime.

    Handles the common MediaInfo shapes: ``UTC 2024-03-15 08:30:00``,
    ``2024-03-15 08:30:00 UTC``, and ISO-8601 variants.

    Args:
        raw (str): The raw track field value.

    Returns:
        Optional[datetime]: Timezone-aware UTC datetime, or None.
    """
    text = raw.strip()
    had_utc_marker = "UTC" in text or text.endswith("Z")
    text = text.replace("UTC", "").replace("Z", "").strip()
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        # MediaInfo dates are UTC unless stated otherwise; an explicit
        # marker or absence both resolve to UTC here.
        dt = dt.replace(tzinfo=timezone.utc)
        if not had_utc_marker:
            logger.debug("Assuming UTC for container date %r", raw)
    return dt.astimezone(timezone.utc)


def probe(video_path: Path) -> Tuple[Optional[datetime],
                                     Optional[Tuple[float, float]]]:
    """Probe a media container for its capture datetime and GPS fix.

    Args:
        video_path (Path): The video file.

    Returns:
        Tuple: ``(datetime | None, (lat, lon) | None)``. Both are None
        whenever pymediainfo/libmediainfo is unavailable, the file
        cannot be parsed, or the fields are absent. Never raises.
    """
    try:
        from pymediainfo import MediaInfo
    except Exception as e:  # noqa: BLE001 - ImportError or missing native lib
        logger.debug("pymediainfo unavailable (%s); no container metadata", e)
        return None, None

    try:
        info = MediaInfo.parse(str(video_path))
    except Exception as e:  # noqa: BLE001 - degrade, never break extraction
        logger.warning("MediaInfo could not parse %s: %s", video_path.name, e)
        return None, None

    dt: Optional[datetime] = None
    gps: Optional[Tuple[float, float]] = None
    for track in info.tracks:
        data = track.to_data()
        if dt is None:
            for key in _DATE_KEYS:
                raw = data.get(key)
                if raw:
                    dt = _parse_date(str(raw))
                    if dt is not None:
                        break
        if gps is None:
            loc = data.get("recorded_location") or data.get("xyz")
            if loc:
                match = _ISO6709.match(str(loc))
                if match:
                    gps = (float(match.group(1)), float(match.group(2)))
        if dt is not None and gps is not None:
            break
    return dt, gps
