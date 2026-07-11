"""Position interpolation over shared GPX telemetry nodes."""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

from .config import IlluminationConfig
from .models import GnssPoint

logger = logging.getLogger(__name__)


class GnssTrack:
    """Interpolates vehicle position at arbitrary timestamps.

    Built from the shared telemetry node list; when none is available
    the configured fixed location answers every query instead.
    """

    def __init__(
        self, config: IlluminationConfig, nodes: Optional[List[dict]] = None,
    ):
        """Index the telemetry nodes, or arm the fixed-location fallback.

        Args:
            config (IlluminationConfig): Validated illumination config.
            nodes (Optional[List[dict]]): Shared GPX telemetry nodes
                with 'coords' as (lon, lat) and aware 'time'; None
                selects the fixed-location fallback.

        Raises:
            ValueError: If nodes are absent and no fixed location is
                configured, or node times are not monotonic.
        """
        self.config = config
        self._times: Optional[np.ndarray] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        if nodes is None:
            if config.fixed_lat is None or config.fixed_lon is None:
                raise ValueError(
                    "illumination requires telemetry or fixed_lat/fixed_lon"
                )
            logger.info("No telemetry: illumination uses the fixed location")
            return
        self._times = np.array(
            [n["time"].timestamp() for n in nodes], dtype=np.float64,
        )
        if np.any(np.diff(self._times) < 0):
            raise ValueError("telemetry track times are not monotonic")
        self._lats = np.array([n["coords"][1] for n in nodes], dtype=np.float64)
        self._lons = np.array([n["coords"][0] for n in nodes], dtype=np.float64)
        logger.info("Indexed telemetry track: %d fixes", len(nodes))

    def position(self, dt: datetime) -> Optional[GnssPoint]:
        """Interpolate the position at a timestamp.

        Args:
            dt (datetime): Timezone-aware timestamp.

        Returns:
            Optional[GnssPoint]: Interpolated fix; the fixed location
                when no track is loaded; None when the timestamp is
                outside the track by more than the tolerance.

        Raises:
            ValueError: If the timestamp is naive.
        """
        if dt.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if self._times is None:
            return GnssPoint(dt, self.config.fixed_lat, self.config.fixed_lon)
        t = dt.timestamp()
        if (t < self._times[0] - self.config.gnss_tolerance_s
                or t > self._times[-1] + self.config.gnss_tolerance_s):
            return None
        lat = float(np.interp(t, self._times, self._lats))
        lon = float(np.interp(t, self._times, self._lons))
        return GnssPoint(dt, lat, lon)