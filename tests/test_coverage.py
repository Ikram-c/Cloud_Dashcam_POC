"""Coverage chunking tests: geometry, graph, intervals, assignment.
Fully offline; GCS mocked via the injectable provider client."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

pytest.importorskip("networkx")
pytest.importorskip("gpxpy")

from frame_extract import Settings
from frame_extract.cloud import MockStorageClient
from frame_extract.coverage import CoverageChunker, CoverageInterval
from frame_extract.geometry import build_bounding_box, liang_barsky
from frame_extract.models import ExtractedFrame
from frame_extract.route_graph import (
    NO_COVERAGE, create_route_dag, subdivide_route_fast,
)
from frame_extract.telemetry import TelemetryProvider

CONFIG = Path(__file__).parent.parent / "config.yaml"
T0 = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)


def _nodes(*points):
    """Build track nodes from (lon, lat, seconds_after_T0) triples."""
    return [
        {"coords": (lon, lat), "time": T0 + timedelta(seconds=s)}
        for lon, lat, s in points
    ]


@pytest.fixture
def settings(tmp_path):
    base = Settings.load(CONFIG)
    telemetry = replace(
        base.telemetry, gpx_path=None,
        gcs_bucket="bucket", gcs_blob="track.gpx", use_mock_gcs=True,
    )
    coverage = replace(
        base.coverage, enabled=True, network="testnet",
        networks={"testnet": {"zone_a": (0.0, 0.0, 10.0, 10.0)}},
    )
    runtime = replace(base.runtime, output_directory=str(tmp_path))
    illumination = replace(base.illumination, timezone="UTC")
    return replace(
        base, telemetry=telemetry, coverage=coverage, runtime=runtime,
        illumination=illumination,
    )


class TestGeometry:
    def test_bbox_orders_coordinates(self):
        assert build_bounding_box((5, 9), (1, 2)) == (1, 2, 5, 9)

    def test_clip_fully_inside(self):
        p1, p2 = liang_barsky(2, 2, 8, 8, 0, 0, 10, 10)
        assert p1 == (2, 2) and p2 == (8, 8)

    def test_clip_crossing(self):
        p1, p2 = liang_barsky(-5, 5, 15, 5, 0, 0, 10, 10)
        assert p1 == (0.0, 5.0) and p2 == (10.0, 5.0)

    def test_miss_returns_none(self):
        assert liang_barsky(20, 20, 30, 30, 0, 0, 10, 10) is None

    def test_inverted_rect_raises(self):
        with pytest.raises(ValueError):
            liang_barsky(0, 0, 1, 1, 10, 0, 0, 10)


class TestRouteGraph:
    def test_dag_requires_two_nodes(self):
        with pytest.raises(ValueError):
            create_route_dag(_nodes((0.0, 0.0, 0)))

    def test_subdivision_labels_crossing(self):
        dag = create_route_dag(_nodes((-5.0, 5.0, 0), (15.0, 5.0, 100)))
        out = subdivide_route_fast(dag, {"z": (0.0, 0.0, 10.0, 10.0)})
        labels = {d["coverage"] for _, _, d in out.edges(data=True)}
        assert "z" in labels
        assert NO_COVERAGE in labels

    def test_untouched_edge_keeps_no_coverage(self):
        dag = create_route_dag(_nodes((20.0, 20.0, 0), (30.0, 20.0, 60)))
        out = subdivide_route_fast(dag, {"z": (0.0, 0.0, 10.0, 10.0)})
        labels = [d["coverage"] for _, _, d in out.edges(data=True)]
        assert all(lab == NO_COVERAGE for lab in labels)


class TestNetworkSelection:
    def test_unknown_network_rejected(self, settings):
        with pytest.raises(ValueError):
            replace(settings.coverage, network="carrier_pigeon")

    def test_zones_property_selects_network(self, settings):
        assert "zone_a" in settings.coverage.zones


class TestIntervals:
    def test_route_through_zone_labels_middle(self, settings):
        """Enter at t=25, exit at t=75 on a straight crossing."""
        chunker = CoverageChunker(settings)
        nodes = _nodes((-5.0, 5.0, 0), (15.0, 5.0, 100))
        intervals = chunker.build_intervals(nodes)
        inside = [iv for iv in intervals if iv.zone == "zone_a"]
        assert len(inside) >= 1
        start = min(iv.start for iv in inside)
        end = max(iv.end for iv in inside)
        assert (end - T0).total_seconds() == pytest.approx(75.0, abs=0.5)
        assert (start - T0).total_seconds() <= 25.5

    def test_route_outside_zone_is_uncovered(self, settings):
        chunker = CoverageChunker(settings)
        nodes = _nodes((20.0, 20.0, 0), (30.0, 20.0, 60))
        intervals = chunker.build_intervals(nodes)
        assert all(iv.zone == NO_COVERAGE for iv in intervals)

    def test_zone_at_outside_track(self):
        intervals = [CoverageInterval(T0, T0 + timedelta(seconds=10), "z")]
        late = T0 + timedelta(hours=1)
        assert CoverageChunker.zone_at(intervals, late) == NO_COVERAGE

    def test_zone_at_naive_raises(self):
        with pytest.raises(ValueError):
            CoverageChunker.zone_at([], datetime(2026, 7, 11, 12, 0))

    def test_interval_end_before_start_raises(self):
        with pytest.raises(ValueError):
            CoverageInterval(T0 + timedelta(seconds=1), T0, "z")


class TestChunking:
    def test_frames_split_by_coverage(self, settings):
        """Frames timed inside/outside the zone land in separate chunks."""
        chunker = CoverageChunker(settings)
        intervals = [
            CoverageInterval(T0, T0 + timedelta(seconds=50), NO_COVERAGE),
            CoverageInterval(
                T0 + timedelta(seconds=50), T0 + timedelta(seconds=100), "zone_a",
            ),
        ]
        video = Path("CAR01_20260711_120000_CAM01.mp4")
        frames = [
            ExtractedFrame(video, i, i / 1.0, Path("f.png"), 100.0, 120.0)
            for i in (10, 20, 60, 70)
        ]
        grouped = chunker._assign(frames, intervals)
        assert {f.frame_index for f in grouped[NO_COVERAGE][video]} == {10, 20}
        assert {f.frame_index for f in grouped["zone_a"][video]} == {60, 70}

    def test_manifest_records_network(self, settings, tmp_path):
        provider = TelemetryProvider(settings, storage_client=MockStorageClient())
        nodes = provider.nodes(tmp_path)
        chunks = CoverageChunker(settings).run([], tmp_path, nodes)
        payload = yaml.safe_load(
            (tmp_path / settings.coverage.chunk_manifest_name).read_text()
        )
        assert payload["network"] == "testnet"
        assert payload["zones"] == {}
        assert chunks == {}


class TestTelemetryProvider:
    def test_mock_gcs_resolves_nodes(self, settings, tmp_path):
        provider = TelemetryProvider(settings, storage_client=MockStorageClient())
        nodes = provider.nodes(tmp_path)
        assert len(nodes) == 3
        assert nodes[0]["time"].tzinfo is not None

    def test_resolves_once(self, settings, tmp_path):
        client = MagicMock()
        client.bucket.side_effect = MockStorageClient().bucket
        provider = TelemetryProvider(settings, storage_client=client)
        provider.nodes(tmp_path)
        provider.nodes(tmp_path)
        assert client.bucket.call_count == 1

    def test_unconfigured_raises(self, settings, tmp_path):
        bare = replace(
            settings, telemetry=replace(
                settings.telemetry, gpx_path=None,
                gcs_bucket=None, gcs_blob=None,
            ),
        )
        provider = TelemetryProvider(bare)
        assert not provider.configured
        with pytest.raises(ValueError):
            provider.nodes(tmp_path)

    def test_local_path_wins_over_gcs(self, settings, tmp_path):
        gpx = tmp_path / "local.gpx"
        gpx.write_text(
            '<?xml version="1.0"?><gpx version="1.1" creator="t"><trk><trkseg>'
            '<trkpt lat="1.0" lon="2.0"><time>2026-07-11T12:00:00Z</time></trkpt>'
            '<trkpt lat="1.1" lon="2.1"><time>2026-07-11T12:01:00Z</time></trkpt>'
            "</trkseg></trk></gpx>",
            encoding="utf-8",
        )
        cfg = replace(
            settings, telemetry=replace(settings.telemetry, gpx_path=str(gpx)),
        )
        provider = TelemetryProvider(cfg)  # no client: must not need one
        nodes = provider.nodes(tmp_path)
        assert len(nodes) == 2
        assert nodes[0]["coords"] == (2.0, 1.0)   # (lon, lat)
