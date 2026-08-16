"""GPX parsing into ordered, timezone-aware track nodes.

Untimed points cannot place the vehicle in time, so they are dropped;
naive timestamps are assumed UTC (the GPX convention).
"""

import logging
from datetime import timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# One track fix: {'coords': (lon, lat), 'time': aware datetime}.
TrackNode = Dict[str, object]


def parse_gpx_to_nodes(file_path: Path) -> List[TrackNode]:
    """Read a GPX file into ordered timestamped nodes.

    Args:
        file_path (Path): Path to the .gpx file.

    Returns:
        List[TrackNode]: Dicts with 'coords' as (lon, lat) and an
            aware 'time', in file order.

    Raises:
        ImportError: If gpxpy is not installed.
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no usable (timed) points.
    """
    try:
        import gpxpy
    except ImportError as e:
        raise ImportError(
            "gpxpy is required for telemetry; install with .[coverage]"
        ) from e
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        gpx = gpxpy.parse(handle)
    nodes: List[TrackNode] = []
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
        logger.warning("%s: dropped %d untimed GPX points", path.name, dropped)
    if not nodes:
        raise ValueError(f"{path}: no usable (timed) GPX points")
    return nodes
