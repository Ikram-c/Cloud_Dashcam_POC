"""Fusion of the sky-model prior with per-frame probe measurements.

A complementary filter on the log-gain state g (observed minus
expected log intensity): probe deltas propagate g between frames
(relative, low-noise), and the full-frame observed mean pulls g
toward an absolute value (noisy, scene-dependent) with weight
``fusion_gain``. Forecast changes enter through the prior itself.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import IlluminationConfig
from .gnss import GnssTrack
from .models import IlluminationPrior, IlluminationState
from .probes import GroundProbe, VehicleProbe
from .sky_model import SkyModel

logger = logging.getLogger(__name__)


class IlluminationEstimator:
    """Per-frame illumination state from prior plus probes."""

    def __init__(
        self, config: IlluminationConfig, track_nodes: Optional[List[dict]] = None,
    ):
        """Wire the sky model, telemetry track, and configured probes.

        Args:
            config (IlluminationConfig): Validated illumination config.
            track_nodes (Optional[List[dict]]): Shared GPX telemetry
                nodes; None selects the fixed-location fallback.
        """
        self.config = config
        self.sky = SkyModel(config)
        self.track = GnssTrack(config, track_nodes)
        self.ground = GroundProbe(config) if config.probe in ("ground", "both") else None
        self.vehicle = (
            VehicleProbe(config) if config.probe in ("vehicle", "both") else None
        )
        self._gain: float = 0.0
        self._initialised = False

    def reset(self):
        """Clear state between videos."""
        self._gain = 0.0
        self._initialised = False
        if self.ground is not None:
            self.ground.reset()
        if self.vehicle is not None:
            self.vehicle.reset()

    def update(
        self, dt: datetime, frame_bgr: np.ndarray
    ) -> Optional[Tuple[IlluminationPrior, IlluminationState]]:
        """Fuse the prior with probe measurements for one frame.

        Args:
            dt (datetime): Timezone-aware frame timestamp.
            frame_bgr (np.ndarray): BGR frame (post-crop).

        Returns:
            Optional[Tuple]: (prior, state), or None when position is
                unavailable for this timestamp.

        Raises:
            ValueError: If the frame is empty.
        """
        if frame_bgr.size == 0:
            raise ValueError("frame must be non-empty")
        fix = self.track.position(dt)
        if fix is None:
            logger.debug("No GNSS fix for %s; illumination skipped", dt)
            return None
        prior = self.sky.prior(dt, fix.lat, fix.lon)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        observed = float(np.mean(np.log(gray.astype(np.float32) + 1.0)))
        absolute = observed - prior.expected_log_intensity

        deltas, sources = [], []
        if self.ground is not None:
            d = self.ground.measure(frame_bgr)
            if d is not None:
                deltas.append(d)
                sources.append("ground_probe")
        if self.vehicle is not None:
            d = self.vehicle.measure(frame_bgr)
            if d is not None:
                deltas.append(d)
                sources.append("vehicle_probe")

        if not self._initialised:
            self._gain = absolute
            self._initialised = True
            source = "prior"
        else:
            propagated = self._gain + (float(np.mean(deltas)) if deltas else 0.0)
            k = self.config.fusion_gain
            self._gain = (1.0 - k) * propagated + k * absolute
            source = "fused" if deltas else "prior"
        confidence = min(
            1.0,
            self.config.confidence_base
            + self.config.confidence_per_probe * len(deltas),
        )
        state = IlluminationState(
            log_gain=self._gain,
            observed_log_mean=observed,
            expected_log_intensity=prior.expected_log_intensity,
            confidence=confidence,
            source=sources[0] if len(sources) == 1 else source,
        )
        return prior, state

    def exposure_band(self, prior: IlluminationPrior) -> Tuple[float, float]:
        """Acceptable observed-log-mean band for the exposure gate.

        Args:
            prior (IlluminationPrior): The frame's prior.

        Returns:
            Tuple[float, float]: (low, high) in log grey units; width
                is the tolerance in photographic stops via ln(2)/stop.
        """
        center = prior.expected_log_intensity + self._gain
        half = self.config.exposure_tolerance_stops * float(np.log(2.0))
        return center - half, center + half

    def normalize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Divide out the current gain estimate in the linear domain.

        Args:
            frame_bgr (np.ndarray): BGR frame.

        Returns:
            np.ndarray: Gain-normalised uint8 frame.
        """
        scale = float(np.exp(-self._gain))
        out = frame_bgr.astype(np.float32) * scale
        return np.clip(out, 0, 255).astype(np.uint8)
