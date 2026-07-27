"""Terrain-agnostic frame quality gates.

Vehicle-mounted cameras share failure modes regardless of platform:
vibration-induced blur, exposure swings, and long static stretches.
The duplicate gate compares log-domain gradient images rather than
intensities (after Weiss, 2001): global exposure change is additive
in log space and vanishes under differentiation, and illumination
edges are sparse, so the median difference stays near zero for a
static scene even under auto-gain or passing shadows.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import QualityConfig
from .models import QualityResult


class FrameQualityGate:
    """Sequential blur, exposure, and duplicate checks."""

    def __init__(self, config: QualityConfig):
        """Initialise the gate.

        Args:
            config (QualityConfig): Validated quality thresholds.
        """
        self.config = config
        self._last_grad: Optional[np.ndarray] = None

    def reset(self):
        """Clear inter-frame state between videos."""
        self._last_grad = None

    def evaluate(
        self,
        frame: np.ndarray,
        exposure_band: Optional[Tuple[float, float]] = None,
    ) -> QualityResult:
        """Run all gates on a frame; first failure short-circuits.

        The duplicate-gate gradient baseline is updated on *every*
        evaluated frame, including ones rejected for blur or exposure,
        so the baseline never goes stale across rejected stretches.

        Args:
            frame (np.ndarray): BGR or grayscale frame (post-crop).
            exposure_band (Optional[Tuple[float, float]]): Dynamic
                (low, high) bounds on the observed *mean log* intensity
                (mean of log(grey + 1), matching the illumination
                estimator's statistic). When None, the static linear
                intensity thresholds apply.

        Returns:
            QualityResult: Acceptance decision and metrics.

        Raises:
            ValueError: If the frame is empty or has invalid dims.
        """
        if frame.size == 0:
            raise ValueError("frame must be non-empty")
        if frame.ndim not in (2, 3):
            raise ValueError("frame must be 2D or 3D")
        small = self._downscale(frame)
        blur_score = self._blur_score(small)
        mean_intensity = float(np.mean(small))
        log_mean = float(np.mean(np.log(small.astype(np.float32) + 1.0)))
        diff = self._diff_from_last(small)
        if not self.config.enabled:
            return QualityResult(True, None, blur_score, mean_intensity, diff)
        if blur_score < self.config.blur_threshold:
            return QualityResult(False, "blur", blur_score, mean_intensity, diff)
        if exposure_band is not None:
            low, high = exposure_band
            if log_mean < low:
                return QualityResult(False, "underexposed", blur_score, mean_intensity, diff)
            if log_mean > high:
                return QualityResult(False, "overexposed", blur_score, mean_intensity, diff)
        else:
            if mean_intensity < self.config.min_mean_intensity:
                return QualityResult(False, "underexposed", blur_score, mean_intensity, diff)
            if mean_intensity > self.config.max_mean_intensity:
                return QualityResult(False, "overexposed", blur_score, mean_intensity, diff)
        if diff is not None and diff < self.config.duplicate_threshold:
            return QualityResult(False, "duplicate", blur_score, mean_intensity, diff)
        return QualityResult(True, None, blur_score, mean_intensity, diff)

    def _downscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale at analysis resolution.

        Args:
            frame (np.ndarray): BGR or grayscale frame.

        Returns:
            np.ndarray: Grayscale frame at the configured size.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, self.config.analysis_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _blur_score(gray: np.ndarray) -> float:
        """Score sharpness as the variance of the Laplacian.

        Args:
            gray (np.ndarray): Grayscale frame.

        Returns:
            float: Laplacian variance; higher is sharper.
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _log_gradient(self, gray: np.ndarray) -> np.ndarray:
        """Compute the summed absolute log-domain gradient image.

        Args:
            gray (np.ndarray): Grayscale frame at analysis size.

        Returns:
            np.ndarray: float32 gradient magnitude image.
        """
        log_g = np.log(gray.astype(np.float32) + 1.0)
        return (
            np.abs(cv2.Sobel(log_g, cv2.CV_32F, 1, 0, ksize=3))
            + np.abs(cv2.Sobel(log_g, cv2.CV_32F, 0, 1, ksize=3))
        )

    def _diff_from_last(self, gray: np.ndarray) -> Optional[float]:
        """Illumination-robust difference against the previous frame.

        Args:
            gray (np.ndarray): Grayscale frame at analysis size.

        Returns:
            Optional[float]: Median abs log-gradient difference against
                the previous *evaluated* frame (kept or not), or None
                on the first frame.
        """
        grad = self._log_gradient(gray)
        if self._last_grad is None:
            self._last_grad = grad
            return None
        diff = float(np.median(np.abs(grad - self._last_grad)))
        self._last_grad = grad
        return diff
