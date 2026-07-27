"""Container-metadata probing tests. MediaInfo mocked; fully offline."""

from datetime import timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from frame_extract import media_metadata

VIDEO = Path("CAR01_20240315_083000_CAM01.mp4")


def _info(**general):
    """Build a fake MediaInfo result with one General track."""
    track = SimpleNamespace(
        track_type="General", encoded_date=None, tagged_date=None,
        recorded_date=None, xyz=None,
    )
    for key, value in general.items():
        setattr(track, key, value)
    other = SimpleNamespace(track_type="Video")
    return SimpleNamespace(tracks=[other, track])


def _patched(fake_parse):
    """Patch the module to behave as if pymediainfo were installed."""
    fake = SimpleNamespace(parse=fake_parse)
    return (
        patch.object(media_metadata, "MEDIAINFO_AVAILABLE", True),
        patch.object(media_metadata, "MediaInfo", fake, create=True),
    )


class TestProbe:
    def test_unavailable_library_degrades_to_none(self):
        with patch.object(media_metadata, "MEDIAINFO_AVAILABLE", False):
            assert media_metadata.probe(VIDEO) == (None, None)

    def test_encoded_date_with_utc_prefix(self):
        p1, p2 = _patched(lambda path: _info(
            encoded_date="UTC 2024-03-15 08:30:00",
        ))
        with p1, p2:
            dt, gps = media_metadata.probe(VIDEO)
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        assert (dt.year, dt.hour, dt.minute) == (2024, 8, 30)
        assert gps is None

    def test_encoded_date_with_utc_suffix_and_t_separator(self):
        p1, p2 = _patched(lambda path: _info(
            encoded_date="2024-03-15T08:30:00 UTC",
        ))
        with p1, p2:
            dt, _ = media_metadata.probe(VIDEO)
        assert dt is not None and dt.hour == 8

    def test_tagged_date_fallback(self):
        p1, p2 = _patched(lambda path: _info(
            tagged_date="2024-03-15 09:00:00",
        ))
        with p1, p2:
            dt, _ = media_metadata.probe(VIDEO)
        assert dt is not None and dt.hour == 9

    def test_xyz_gps_parsed(self):
        p1, p2 = _patched(lambda path: _info(
            encoded_date="UTC 2024-03-15 08:30:00",
            xyz="+52.4500+001.7300/",
        ))
        with p1, p2:
            _, gps = media_metadata.probe(VIDEO)
        assert gps is not None
        lat, lon = gps
        assert lat == pytest.approx(52.45)
        assert lon == pytest.approx(1.73)

    def test_unparseable_date_returns_none(self):
        p1, p2 = _patched(lambda path: _info(encoded_date="not a date"))
        with p1, p2:
            dt, _ = media_metadata.probe(VIDEO)
        assert dt is None

    def test_parse_failure_degrades_to_none(self):
        def broken(path):
            raise RuntimeError("corrupt container")
        p1, p2 = _patched(broken)
        with p1, p2:
            assert media_metadata.probe(VIDEO) == (None, None)

    def test_no_tags_returns_none(self):
        p1, p2 = _patched(lambda path: _info())
        with p1, p2:
            assert media_metadata.probe(VIDEO) == (None, None)
