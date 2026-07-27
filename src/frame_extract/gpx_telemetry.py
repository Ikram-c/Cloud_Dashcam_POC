"""GPX parsing into the shared telemetry node list.

Reconstructed module: the original was lost from the repository.
Nodes carry 'coords' as (lon, lat) and a timezone-aware 'time';
untimed points are dropped and naive times are assumed UTC, per the
documented behaviour.
"""

import logging
from pathlib import Path
from typing import List

from datetime import timezone

logger = logging.getLogger(__name__)


def parse_gpx_to_nodes(gpx_path: Path) -> List[dict]:
    """Parse a GPX file into ordered telemetry nodes.

    Args:
        gpx_path (Path): Path to the .gpx file.

    Returns:
        List[dict]: Nodes with 'coords' (lon, lat) and aware 'time',
            in track order.

    Raises:
        ImportError: If gpxpy is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the track contains no timed points.
    """
    import gpxpy  # deferred: only the coverage/illumination path needs it

    gpx_path = Path(gpx_path)
    if not gpx_path.exists():
        raise FileNotFoundError(f"GPX file not found: {gpx_path}")
    with gpx_path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)
    nodes: List[dict] = []
    dropped = 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is None:
                    dropped += 1
                    continue
                time = point.time
                if time.tzinfo is None:
                    time = time.replace(tzinfo=timezone.utc)
                nodes.append({
                    "coords": (point.longitude, point.latitude),
                    "time": time,
                })
    if dropped:
        logger.warning("%s: dropped %d untimed track points", gpx_path.name, dropped)
    if not nodes:
        raise ValueError(f"GPX track has no timed points: {gpx_path}")
    logger.info("Parsed %d telemetry nodes from %s", len(nodes), gpx_path.name)
    return nodes
