"""Solar geometry and forecast-based illumination priors.

Sun position uses the NOAA analytical approximation: exact enough
here and requires no network. Cloud cover and shortwave irradiance
come from an hourly forecast provider and modulate the clear-sky
expectation. All model constants are configuration values so that
scripts/calibrate_sky_model.py can fit them to a specific camera.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import requests

from .config import IlluminationConfig
from .models import IlluminationPrior

logger = logging.getLogger(__name__)


def solar_position(dt: datetime, lat: float, lon: float) -> Tuple[float, float]:
    """Compute sun elevation and azimuth (NOAA approximation).

    Args:
        dt (datetime): Timezone-aware timestamp.
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        Tuple[float, float]: (elevation_deg, azimuth_deg).

    Raises:
        ValueError: If the timestamp is naive or coordinates invalid.
    """
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError("coordinates out of range")
    dt = dt.astimezone(timezone.utc)
    day = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma)
    )
    tst = hour * 60.0 + eqtime + 4.0 * lon
    ha = math.radians(tst / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zen = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    )
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    elevation = 90.0 - math.degrees(zen)
    azimuth = math.degrees(math.atan2(
        -math.sin(ha),
        math.tan(decl) * math.cos(lat_r) - math.sin(lat_r) * math.cos(ha),
    )) % 360.0
    return elevation, azimuth


class SkyModel:
    """Produces IlluminationPrior objects for (time, position) pairs."""

    def __init__(self, config: IlluminationConfig):
        """Initialise the model and load the forecast cache.

        Args:
            config (IlluminationConfig): Validated illumination config.
        """
        self.config = config
        self._cache: dict = self._load_cache()
        self._forecast_warned = False
        self.session = requests.Session()

    def prior(self, dt: datetime, lat: float, lon: float) -> IlluminationPrior:
        """Build the illumination prior for one timestamp and place.

        Args:
            dt (datetime): Timezone-aware timestamp.
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            IlluminationPrior: Geometry plus forecast when available.
        """
        elevation, azimuth = solar_position(dt, lat, lon)
        daylight = elevation > self.config.twilight_elevation_deg
        cloud, swr = self._forecast(dt, lat, lon)
        expected = self.expected_log_intensity(elevation, cloud)
        return IlluminationPrior(
            timestamp=dt, sun_elevation_deg=elevation, sun_azimuth_deg=azimuth,
            cloud_fraction=cloud, shortwave_wm2=swr,
            expected_log_intensity=expected, daylight=daylight,
        )

    def expected_log_intensity(self, elevation: float, cloud: Optional[float]) -> float:
        """Predict the mean log grey level from geometry and cloud.

        Args:
            elevation (float): Sun elevation in degrees.
            cloud (Optional[float]): Cloud fraction in [0, 1], or None.

        Returns:
            float: Expected mean of log(grey + 1).
        """
        c = self.config
        if elevation <= c.twilight_elevation_deg:
            return c.night_log_intensity
        sun_factor = math.sin(math.radians(max(elevation, 0.0)))
        base = c.night_log_intensity + (
            c.clear_sky_log_intensity - c.night_log_intensity
        ) * min(1.0, c.min_sun_factor + sun_factor)
        if cloud is not None:
            base -= c.cloud_attenuation_log * cloud
        return base

    def _forecast(
        self, dt: datetime, lat: float, lon: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Fetch hourly cloud cover and irradiance, with a disk cache.

        Historical timestamps (older than ``forecast_max_past_days``)
        are routed to the archive endpoint; the forecast endpoint does
        not serve them, which previously made the cloud term silently
        vanish for exactly the after-the-fact footage this pipeline
        usually processes.

        Args:
            dt (datetime): Timezone-aware timestamp.
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            Tuple[Optional[float], Optional[float]]: (cloud fraction,
                shortwave W/m2); (None, None) when disabled or failed,
                in which case the prior is geometry-only.
        """
        if self.config.forecast_provider != "open_meteo":
            return None, None
        dt_utc = dt.astimezone(timezone.utc)
        key = f"{round(lat, 2)},{round(lon, 2)},{dt_utc.strftime('%Y-%m-%dT%H')}"
        if key in self._cache:
            entry = self._cache[key]
            return entry["cloud"], entry["swr"]
        age_days = (datetime.now(timezone.utc) - dt_utc).days
        url = (
            self.config.forecast_archive_url
            if age_days > self.config.forecast_max_past_days
            else self.config.forecast_url
        )
        try:
            resp = self.session.get(
                url,
                params={
                    "latitude": lat, "longitude": lon,
                    "hourly": "cloud_cover,shortwave_radiation",
                    "start_date": dt_utc.strftime("%Y-%m-%d"),
                    "end_date": dt_utc.strftime("%Y-%m-%d"),
                    "timezone": "UTC",
                },
                timeout=self.config.http_timeout_s,
            )
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            idx = dt_utc.hour
            cloud = float(hourly["cloud_cover"][idx]) / 100.0
            swr = float(hourly["shortwave_radiation"][idx])
        except (requests.RequestException, KeyError, ValueError, IndexError) as e:
            if not self._forecast_warned:
                logger.warning(
                    "Forecast unavailable (%s); priors are geometry-only "
                    "(cloud attenuation not applied). Further failures "
                    "logged at debug level.", e,
                )
                self._forecast_warned = True
            else:
                logger.debug("Forecast unavailable (%s); geometry-only prior", e)
            return None, None
        self._cache[key] = {"cloud": cloud, "swr": swr}
        self._save_cache()
        return cloud, swr

    def _load_cache(self) -> dict:
        """Load the forecast cache, tolerating corruption.

        Returns:
            dict: Cached forecast entries.
        """
        path = Path(self.config.cache_path)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError):
                return {}
        return {}

    def _save_cache(self):
        """Persist the cache atomically, merging concurrent writers.

        The on-disk cache is re-read and merged before writing so that
        parallel workers extend rather than overwrite each other; the
        final rename is atomic. A race can still drop an entry written
        between the merge and the rename - harmless, it just re-fetches.
        """
        path = Path(self.config.cache_path)
        merged = {**self._load_cache(), **self._cache}
        self._cache = merged
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(merged))
            os.replace(tmp, path)
        except OSError as e:
            logger.debug("Cache write failed: %s", e)