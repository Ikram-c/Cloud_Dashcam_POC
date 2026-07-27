"""Weiss (2001) reflectance recovery tests. Fully offline."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frame_extract import Settings
from frame_extract.intrinsic import WeissReflectanceEstimator

CONFIG = Path(__file__).parent.parent / "config.yaml"

H, W = 96, 128


@pytest.fixture
def config():
    return Settings.load(CONFIG).intrinsic


def _static_scene() -> np.ndarray:
    """A fixed textured reflectance image."""
    rng = np.random.default_rng(11)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 120.0 + 40.0 * np.sin(0.1 * x) * np.cos(0.13 * y)
    base += rng.normal(0.0, 5.0, base.shape)
    frame = np.clip(base, 20, 235).astype(np.uint8)
    return np.stack([frame] * 3, axis=2)


def _with_shadow(scene: np.ndarray, left: int, width: int) -> np.ndarray:
    """Apply a multiplicative shadow band starting at column ``left``."""
    out = scene.astype(np.float32)
    out[:, left:left + width] *= 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


class TestWeissEstimator:
    def test_min_frames_guard(self, config):
        est = WeissReflectanceEstimator(config)
        scene = _static_scene()
        for _ in range(config.min_frames - 1):
            est.add_frame(scene)
        assert not est.ready
        assert est.estimate_reflectance() is None

    def test_buffer_bound_enforced(self, config):
        small = replace(config, min_frames=3, max_frames=4)
        est = WeissReflectanceEstimator(small)
        scene = _static_scene()
        results = [est.add_frame(scene) for _ in range(6)]
        assert results == [True, True, True, True, False, False]
        assert est.frame_count == 4

    def test_shape_mismatch_raises(self, config):
        est = WeissReflectanceEstimator(config)
        est.add_frame(_static_scene())
        with pytest.raises(ValueError):
            est.add_frame(np.zeros((H + 2, W, 3), dtype=np.uint8))

    def test_translating_shadow_suppressed(self, config):
        """Weiss Fig. 5 setup: constant reflectance, moving shadow.

        The recovered reflectance should be near-uniform in mean
        between the region every frame shadowed differently and a
        never-shadowed reference strip.
        """
        est = WeissReflectanceEstimator(config)
        scene = _static_scene()
        band = 24
        for i in range(9):
            est.add_frame(_with_shadow(scene, left=8 + i * 10, width=band))
        reflectance = est.estimate_reflectance()
        assert reflectance is not None
        gray = reflectance.mean(axis=2)
        clean = gray[:, W - 16:].mean()          # never shadowed
        swept = gray[:, 20:80].mean()            # shadowed in some frames
        original = scene.mean(axis=2)
        naive_swept = _with_shadow(scene, 30, band).mean(axis=2)[:, 20:80].mean()
        recovered_err = abs(swept - original[:, 20:80].mean())
        naive_err = abs(naive_swept - original[:, 20:80].mean())
        assert recovered_err < naive_err * 0.5
        assert abs(swept - clean) < 25.0

    def test_reset_clears_state(self, config):
        est = WeissReflectanceEstimator(config)
        est.add_frame(_static_scene())
        est.reset()
        assert est.frame_count == 0
        assert est.add_frame(np.zeros((H + 2, W, 3), dtype=np.uint8))
