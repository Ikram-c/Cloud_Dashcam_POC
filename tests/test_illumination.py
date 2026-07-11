"""Sky model, fusion, GNSS, and telemetry-track tests. Fully offline."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import requests

from frame_extract import Settings
from frame_extract.gnss import GnssTrack
from frame_extract.illumination import IlluminationEstimator
from frame_extract.sky_model import SkyModel, solar_position

CONFIG = Path(__file__).parent.parent / "config.yaml"


def _track_nodes(*points):
    """(lon, lat, iso_utc_string) triples to telemetry nodes."""
    return [
        {"coords": (lon, lat),
         "time": datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)}
        for lon, lat, iso in points
    ]


@pytest.fixture
def illum_config(tmp_path):
    base = Settings.load(CONFIG).illumination
    return replace(
        base, forecast_provider="none",
        cache_path=str(tmp_path / "cache.json"),
        probe="ground",
    )


class TestSolarPosition:
    def test_greenwich_summer_solstice(self):
        """Solar noon at Greenwich, 2024-06-20: elevation ~62 deg."""
        dt = datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc)
        elevation, azimuth = solar_position(dt, 51.4769, 0.0)
        assert elevation == pytest.approx(61.9, abs=1.0)
        assert azimuth == pytest.approx(180.0, abs=8.0)

    def test_midnight_below_horizon(self):
        dt = datetime(2024, 6, 20, 0, 0, tzinfo=timezone.utc)
        elevation, _ = solar_position(dt, 51.4769, 0.0)
        assert elevation < 0.0

    def test_naive_timestamp_raises(self):
        with pytest.raises(ValueError):
            solar_position(datetime(2024, 6, 20, 12, 0), 51.0, 0.0)

    def test_bad_coordinates_raise(self):
        dt = datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            solar_position(dt, 200.0, 0.0)


class TestSkyModel:
    def test_night_returns_floor(self, illum_config):
        sky = SkyModel(illum_config)
        value = sky.expected_log_intensity(-20.0, None)
        assert value == illum_config.night_log_intensity

    def test_cloud_dims_expectation(self, illum_config):
        sky = SkyModel(illum_config)
        clear = sky.expected_log_intensity(45.0, 0.0)
        overcast = sky.expected_log_intensity(45.0, 1.0)
        assert overcast == pytest.approx(
            clear - illum_config.cloud_attenuation_log
        )

    def test_forecast_failure_degrades_to_geometry(self, illum_config):
        config = replace(illum_config, forecast_provider="open_meteo")
        sky = SkyModel(config)
        with patch.object(
            sky.session, "get",
            side_effect=requests.ConnectionError("offline"),
        ):
            dt = datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc)
            prior = sky.prior(dt, 52.0, 1.0)
        assert prior.cloud_fraction is None
        assert prior.expected_log_intensity > 0.0

    def test_historical_date_uses_archive_endpoint(self, illum_config):
        config = replace(illum_config, forecast_provider="open_meteo")
        sky = SkyModel(config)
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            raise requests.ConnectionError("offline")

        with patch.object(sky.session, "get", side_effect=fake_get):
            dt = datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc)
            sky.prior(dt, 52.0, 1.0)
        assert captured["url"] == config.forecast_archive_url


class TestGnssTrack:
    def test_fixed_location_without_track(self, illum_config):
        track = GnssTrack(illum_config, nodes=None)
        dt = datetime(2024, 3, 15, 9, 0, tzinfo=timezone.utc)
        assert track.position(dt).lat == illum_config.fixed_lat

    def test_missing_fixed_and_track_raises(self, illum_config):
        bare = replace(illum_config, fixed_lat=None, fixed_lon=None)
        with pytest.raises(ValueError):
            GnssTrack(bare, nodes=None)

    def test_interpolation_midpoint(self, illum_config):
        nodes = _track_nodes(
            (1.0, 52.0, "2024-03-15T08:00:00"),
            (2.0, 53.0, "2024-03-15T09:00:00"),
        )
        track = GnssTrack(illum_config, nodes)
        fix = track.position(datetime(2024, 3, 15, 8, 30, tzinfo=timezone.utc))
        assert fix.lat == pytest.approx(52.5)
        assert fix.lon == pytest.approx(1.5)

    def test_outside_tolerance_returns_none(self, illum_config):
        nodes = _track_nodes((1.0, 52.0, "2024-03-15T08:00:00"))
        track = GnssTrack(illum_config, nodes)
        assert track.position(
            datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
        ) is None

    def test_non_monotonic_track_raises(self, illum_config):
        nodes = _track_nodes(
            (1.0, 52.0, "2024-03-15T09:00:00"),
            (2.0, 53.0, "2024-03-15T08:00:00"),
        )
        with pytest.raises(ValueError):
            GnssTrack(illum_config, nodes)

    def test_naive_query_raises(self, illum_config):
        track = GnssTrack(illum_config, nodes=None)
        with pytest.raises(ValueError):
            track.position(datetime(2024, 3, 15, 9, 0))


class TestFusion:
    def test_gain_converges_to_constant_offset(self, illum_config):
        """A frame consistently brighter than the prior converges g."""
        estimator = IlluminationEstimator(illum_config)
        rng = np.random.default_rng(3)
        frame = rng.integers(150, 200, (240, 320, 3), dtype=np.uint8)
        dt = datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc)
        gains = []
        for _ in range(20):
            _, state = estimator.update(dt, frame)
            gains.append(state.log_gain)
        assert abs(gains[-1] - gains[-2]) < 1e-3

    def test_night_band_accepts_dark_frame(self, illum_config):
        estimator = IlluminationEstimator(illum_config)
        dark = np.full((240, 320, 3), 12, dtype=np.uint8)
        dt = datetime(2024, 6, 20, 0, 30, tzinfo=timezone.utc)
        prior, _ = estimator.update(dt, dark)
        low, high = estimator.exposure_band(prior)
        observed = float(np.log(12.0 + 1.0))
        assert low <= observed <= high

    def test_estimator_uses_telemetry_nodes(self, illum_config):
        nodes = _track_nodes(
            (1.0, 52.0, "2024-06-20T11:00:00"),
            (1.0, 52.0, "2024-06-20T13:00:00"),
        )
        estimator = IlluminationEstimator(illum_config, nodes)
        frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        inside = estimator.update(
            datetime(2024, 6, 20, 12, 0, tzinfo=timezone.utc), frame,
        )
        outside = estimator.update(
            datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc), frame,
        )
        assert inside is not None
        assert outside is None

    def test_normalize_inverts_gain(self, illum_config):
        estimator = IlluminationEstimator(illum_config)
        estimator._gain = float(np.log(2.0))
        estimator._initialised = True
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        out = estimator.normalize(frame)
        assert float(np.mean(out)) == pytest.approx(50.0, abs=1.0)