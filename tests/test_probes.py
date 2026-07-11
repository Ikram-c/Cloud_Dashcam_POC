"""GroundProbe and VehicleProbe tests. Detector mocked; fully offline."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from frame_extract import Settings
from frame_extract.probes import GroundProbe, VehicleProbe

CONFIG = Path(__file__).parent.parent / "config.yaml"


def _flat(value: int) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


def _detections(*rows) -> np.ndarray:
    """Build an SSD output array from (cls, conf, x1, y1, x2, y2) rows
    with normalised coordinates."""
    out = np.zeros((1, 1, max(len(rows), 1), 7), dtype=np.float32)
    for i, (cls, conf, x1, y1, x2, y2) in enumerate(rows):
        out[0, 0, i] = [0.0, cls, conf, x1, y1, x2, y2]
    return out


@pytest.fixture
def illum_config(tmp_path):
    base = Settings.load(CONFIG).illumination
    return replace(
        base,
        forecast_provider="none",
        cache_path=str(tmp_path / "cache.json"),
        vehicle_confidence=0.5,
        vehicle_iou_min=0.3,
        vehicle_min_track_frames=2,
    )


@pytest.fixture
def vehicle_probe(illum_config, tmp_path):
    """A VehicleProbe with real files on disk and a mocked DNN."""
    proto = tmp_path / "m.prototxt"
    weights = tmp_path / "m.caffemodel"
    proto.touch()
    weights.touch()
    config = replace(
        illum_config,
        vehicle_model_prototxt=str(proto),
        vehicle_model_weights=str(weights),
    )
    net = MagicMock()
    with patch("frame_extract.probes.cv2.dnn.readNetFromCaffe", return_value=net):
        probe = VehicleProbe(config)
    return probe, net


class TestGroundProbe:
    def test_first_frame_returns_none(self, illum_config):
        assert GroundProbe(illum_config).measure(_flat(100)) is None

    def test_delta_is_log_ratio(self, illum_config):
        probe = GroundProbe(illum_config)
        probe.measure(_flat(100))
        delta = probe.measure(_flat(201))
        assert delta == pytest.approx(np.log(202.0 / 101.0), abs=1e-3)

    def test_reset_clears_baseline(self, illum_config):
        probe = GroundProbe(illum_config)
        probe.measure(_flat(100))
        probe.reset()
        assert probe.measure(_flat(100)) is None

    def test_empty_frame_raises(self, illum_config):
        with pytest.raises(ValueError):
            GroundProbe(illum_config).measure(np.empty((0, 0, 3), dtype=np.uint8))


class TestVehicleProbeAvailability:
    def test_disabled_without_paths(self, illum_config):
        probe = VehicleProbe(replace(
            illum_config, vehicle_model_prototxt=None, vehicle_model_weights=None,
        ))
        assert not probe.available
        assert probe.measure(_flat(100)) is None

    def test_disabled_when_files_missing(self, illum_config):
        probe = VehicleProbe(replace(
            illum_config,
            vehicle_model_prototxt="no/such.prototxt",
            vehicle_model_weights="no/such.caffemodel",
        ))
        assert not probe.available


class TestVehicleProbeTracking:
    CAR_BOX = (7, 0.9, 0.4, 0.4, 0.6, 0.6)  # class 7 = car (VOC)

    def test_first_frames_return_none_then_delta(self, vehicle_probe):
        probe, net = vehicle_probe
        net.forward.return_value = _detections(self.CAR_BOX)
        assert probe.measure(_flat(100)) is None        # seed, track_len=1
        delta = probe.measure(_flat(201))               # track_len=2 == min
        assert delta == pytest.approx(np.log(202.0 / 101.0), abs=0.02)

    def test_low_confidence_ignored(self, vehicle_probe):
        probe, net = vehicle_probe
        net.forward.return_value = _detections((7, 0.2, 0.4, 0.4, 0.6, 0.6))
        assert probe.measure(_flat(100)) is None
        assert probe._track_box is None

    def test_non_vehicle_class_ignored(self, vehicle_probe):
        probe, net = vehicle_probe
        net.forward.return_value = _detections((15, 0.95, 0.4, 0.4, 0.6, 0.6))
        assert probe.measure(_flat(100)) is None
        assert probe._track_box is None

    def test_track_loss_resets_baseline(self, vehicle_probe):
        """A jump with IoU below the minimum must re-seed and suppress
        the delta until the new track stabilises."""
        probe, net = vehicle_probe
        net.forward.return_value = _detections(self.CAR_BOX)
        probe.measure(_flat(100))
        probe.measure(_flat(100))
        net.forward.return_value = _detections((7, 0.9, 0.0, 0.0, 0.15, 0.15))
        assert probe.measure(_flat(201)) is None

    def test_no_detection_resets_track(self, vehicle_probe):
        probe, net = vehicle_probe
        net.forward.return_value = _detections(self.CAR_BOX)
        probe.measure(_flat(100))
        net.forward.return_value = _detections()  # empty
        assert probe.measure(_flat(100)) is None
        assert probe._track_box is None

    def test_confidence_sort_regression(self, vehicle_probe):
        """The best valid detection placed at index 11 (past the old
        first-10 slice) must still be selected under explicit sorting."""
        probe, net = vehicle_probe
        noise = [(15, 0.99, 0.0, 0.0, 0.1, 0.1)] * 11   # invalid class
        rows = noise + [self.CAR_BOX]
        net.forward.return_value = _detections(*rows)
        probe.measure(_flat(100))
        assert probe._track_box is not None
        x, y, w, h = probe._track_box
        assert (x, y) == (int(0.4 * 320), int(0.4 * 240))


class TestIoU:
    def test_identical_boxes(self):
        assert VehicleProbe._iou((10, 10, 20, 20), (10, 10, 20, 20)) == 1.0

    def test_disjoint_boxes(self):
        assert VehicleProbe._iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0

    def test_half_overlap(self):
        # (0,0,20,10) vs (10,0,20,10): inter=100, union=300
        assert VehicleProbe._iou((0, 0, 20, 10), (10, 0, 20, 10)) == pytest.approx(1 / 3)