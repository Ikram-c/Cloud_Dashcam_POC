"""Weiss (2001) intrinsic-image recovery over static segments.

Runs only on frames the duplicate gate buffered (its stationarity
assumption). Illumination edges are sparse and transient, so the
temporal median of log-gradient fields isolates the constant
reflectance; a Fourier pseudo-inverse solves the resulting Poisson
problem, and the DC term is restored from the temporal median image.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .config import IntrinsicConfig

logger = logging.getLogger(__name__)


class WeissReflectanceEstimator:
    """Buffers one static segment and recovers its reflectance."""

    def __init__(self, config: IntrinsicConfig):
        """Store the configuration and start with an empty buffer.

        Args:
            config (IntrinsicConfig): Segment bounds and log epsilon.
        """
        self.config = config
        self._logs: List[np.ndarray] = []
        self._shape: Optional[Tuple[int, ...]] = None

    @property
    def frame_count(self) -> int:
        """Number of frames currently buffered."""
        return len(self._logs)

    @property
    def ready(self) -> bool:
        """Whether enough frames are buffered for a median estimate."""
        return len(self._logs) >= self.config.min_frames

    def reset(self):
        """Clear the buffer between static segments."""
        self._logs.clear()
        self._shape = None

    def add_frame(self, frame: np.ndarray) -> bool:
        """Buffer one frame of the current static segment.

        Args:
            frame (np.ndarray): uint8 frame, (H, W) or (H, W, C).

        Returns:
            bool: True if buffered; False when the buffer is already
                at max_frames (bounded memory, NASA rule 2).

        Raises:
            ValueError: If the shape differs from earlier frames.
        """
        if self._shape is not None and frame.shape != self._shape:
            raise ValueError(
                f"frame shape {frame.shape} != segment shape {self._shape}"
            )
        if len(self._logs) >= self.config.max_frames:
            return False
        self._shape = frame.shape
        self._logs.append(
            np.log(frame.astype(np.float32) + self.config.log_epsilon)
        )
        return True

    def estimate_reflectance(self) -> Optional[np.ndarray]:
        """Recover the reflectance image from the buffered segment.

        Returns:
            Optional[np.ndarray]: uint8 reflectance image with the
                buffered shape, or None when below min_frames.
        """
        if not self.ready:
            return None
        stack = np.stack(self._logs)                    # (T, H, W[, C])
        squeeze = stack.ndim == 3
        if squeeze:
            stack = stack[..., np.newaxis]              # (T, H, W, 1)
        # Circular forward differences along x (width) and y (height).
        dx = np.roll(stack, -1, axis=2) - stack
        dy = np.roll(stack, -1, axis=1) - stack
        median_dx = np.median(dx, axis=0)               # (H, W, C)
        median_dy = np.median(dy, axis=0)
        median_log = np.median(stack, axis=0)           # DC reference
        height, width = median_dx.shape[:2]
        denom, conj_hx, conj_hy = self._filters(height, width)
        channels = []
        for c in range(median_dx.shape[2]):
            spectrum = (
                conj_hx * np.fft.fft2(median_dx[:, :, c])
                + conj_hy * np.fft.fft2(median_dy[:, :, c])
            )
            spectrum = np.divide(
                spectrum, denom, out=np.zeros_like(spectrum), where=denom > 0,
            )
            log_r = np.real(np.fft.ifft2(spectrum))
            log_r += float(median_log[:, :, c].mean()) - float(log_r.mean())
            channels.append(log_r)
        log_reflectance = np.stack(channels, axis=2)
        reflectance = np.exp(log_reflectance) - self.config.log_epsilon
        reflectance = np.clip(reflectance, 0.0, 255.0).astype(np.uint8)
        if squeeze:
            reflectance = reflectance[:, :, 0]
        logger.debug(
            "Weiss reflectance from %d frames (%dx%d)",
            self.frame_count, width, height,
        )
        return reflectance

    @staticmethod
    def _filters(height: int, width: int):
        """Frequency responses for the circular difference operators.

        The forward difference d[n] = I[n+1] - I[n] has DFT response
        h[k] = exp(2*pi*i*k/N) - 1; the least-squares Poisson solve is
        the pseudo-inverse (conj(hx)*DX + conj(hy)*DY) / (|hx|^2+|hy|^2)
        with the zero-frequency term handled separately (DC restore).

        Args:
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple: (denominator, conj_hx, conj_hy) broadcastable to
                an (height, width) spectrum.
        """
        fx = np.exp(2j * np.pi * np.fft.fftfreq(width)) - 1.0     # (W,)
        fy = np.exp(2j * np.pi * np.fft.fftfreq(height)) - 1.0    # (H,)
        conj_hx = np.conj(fx)[np.newaxis, :]
        conj_hy = np.conj(fy)[:, np.newaxis]
        denom = (np.abs(fx) ** 2)[np.newaxis, :] + (np.abs(fy) ** 2)[:, np.newaxis]
        return denom, conj_hx, conj_hy
