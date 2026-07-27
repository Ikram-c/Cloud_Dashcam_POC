"""Reflectance recovery over static segments, after Weiss (2001).

Given T frames with constant reflectance and varying illumination,
the ML estimate of the reflectance image under a Laplacian prior on
illumination derivative-filter outputs is: filter each log frame
with horizontal and vertical derivative filters, take the pixelwise
temporal median of each filter output, then invert the filtering via
the pseudo-inverse (a Poisson-type reconstruction in the Fourier
domain).

Reference:
    Weiss, Y. (2001). "Deriving intrinsic images from image
    sequences." Proceedings of ICCV 2001.

Validity: the estimator assumes a stationary camera and scene. It is
applied only to static segments detected by the duplicate gate; it
must not be applied to moving footage.
"""

import logging
from typing import List, Optional

import numpy as np

from .config import IntrinsicConfig

logger = logging.getLogger(__name__)


class WeissReflectanceEstimator:
    """Accumulates a static segment and emits one reflectance frame."""

    def __init__(self, config: IntrinsicConfig):
        """Initialise the estimator.

        Args:
            config (IntrinsicConfig): Validated intrinsic parameters.
        """
        self.config = config
        self._log_frames: List[np.ndarray] = []
        self._shape: Optional[tuple] = None

    def reset(self):
        """Discard the current segment buffer."""
        self._log_frames = []
        self._shape = None

    @property
    def frame_count(self) -> int:
        """Number of frames currently buffered.

        Returns:
            int: Buffer length.
        """
        return len(self._log_frames)

    @property
    def ready(self) -> bool:
        """Whether enough frames are buffered for a reliable estimate.

        Returns:
            bool: True if the minimum segment length is met.
        """
        return len(self._log_frames) >= self.config.min_frames

    def add_frame(self, frame_bgr: np.ndarray) -> bool:
        """Buffer one frame of a static segment.

        Args:
            frame_bgr (np.ndarray): BGR frame (post-crop).

        Returns:
            bool: True if buffered; False if the buffer is full.

        Raises:
            ValueError: If the frame is empty or shape-inconsistent.
        """
        if frame_bgr.size == 0:
            raise ValueError("frame must be non-empty")
        if self._shape is not None and frame_bgr.shape != self._shape:
            raise ValueError("all frames in a segment must share a shape")
        if len(self._log_frames) >= self.config.max_frames:
            return False
        self._shape = frame_bgr.shape
        self._log_frames.append(
            np.log(frame_bgr.astype(np.float32) + self.config.log_epsilon)
        )
        return True

    def estimate_reflectance(self) -> Optional[np.ndarray]:
        """Compute the ML reflectance image for the buffered segment.

        Per channel: median over time of horizontal and vertical
        log-gradients, then Fourier-domain pseudo-inverse
        reconstruction; the DC level lost under differentiation is
        restored from the temporal-median image. (The temporal mean
        is biased dark by transient shadows; the per-pixel temporal
        median is not, provided each pixel is shadowed in a minority
        of the buffered frames - the same sparse-illumination
        assumption the gradient median already relies on.)

        Returns:
            Optional[np.ndarray]: uint8 BGR reflectance image, or
                None if fewer than min_frames are buffered.
        """
        if not self.ready:
            logger.debug(
                "Segment too short for reflectance (%d/%d)",
                len(self._log_frames), self.config.min_frames,
            )
            return None
        stack = np.stack(self._log_frames, axis=0)
        channels = [
            self._estimate_channel(stack[:, :, :, c]) for c in range(stack.shape[3])
        ]
        log_r = np.stack(channels, axis=2)
        reflectance = np.exp(log_r) - self.config.log_epsilon
        return np.clip(reflectance, 0, 255).astype(np.uint8)

    @staticmethod
    def _estimate_channel(log_stack: np.ndarray) -> np.ndarray:
        """Recover one log-reflectance channel.

        Args:
            log_stack (np.ndarray): (T, H, W) log frames.

        Returns:
            np.ndarray: (H, W) log-reflectance channel.
        """
        dx = log_stack[:, :, 1:] - log_stack[:, :, :-1]
        dy = log_stack[:, 1:, :] - log_stack[:, :-1, :]
        med_dx = np.median(dx, axis=0)
        med_dy = np.median(dy, axis=0)
        h, w = log_stack.shape[1:]
        gx = np.zeros((h, w), dtype=np.float32)
        gy = np.zeros((h, w), dtype=np.float32)
        gx[:, 1:] = med_dx
        gy[1:, :] = med_dy
        log_r = WeissReflectanceEstimator._poisson_solve(gx, gy)
        anchor = float(np.mean(np.median(log_stack, axis=0)))
        log_r += anchor - float(np.mean(log_r))
        return log_r

    @staticmethod
    def _poisson_solve(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Pseudo-inverse reconstruction from median gradients.

        Solves Weiss's equation (6) in the Fourier domain: the image
        whose forward-difference gradients best match (gx, gy) in the
        least-squares sense.

        Note: the FFT inversion assumes periodic image boundaries, so
        non-periodic scenes acquire a small seam bias at the image
        edges. Acceptable for Weiss reconstruction; do not rely on
        edge rows/columns of the output.

        Args:
            gx (np.ndarray): Median horizontal log-gradient field.
            gy (np.ndarray): Median vertical log-gradient field.

        Returns:
            np.ndarray: Reconstructed log image (zero-mean DC).
        """
        h, w = gx.shape
        fx = np.zeros((h, w), dtype=np.float32)
        fy = np.zeros((h, w), dtype=np.float32)
        fx[0, 0], fx[0, -1] = 1.0, -1.0
        fy[0, 0], fy[-1, 0] = 1.0, -1.0
        fx_f = np.fft.fft2(fx)
        fy_f = np.fft.fft2(fy)
        denom = np.abs(fx_f) ** 2 + np.abs(fy_f) ** 2
        denom[0, 0] = 1.0
        numer = np.conj(fx_f) * np.fft.fft2(gx) + np.conj(fy_f) * np.fft.fft2(gy)
        r_f = numer / denom
        r_f[0, 0] = 0.0
        return np.real(np.fft.ifft2(r_f)).astype(np.float32)
