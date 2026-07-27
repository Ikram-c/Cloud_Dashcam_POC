"""Per-frame illumination probes.

Ground probe: median log intensity of a fixed ground patch ahead of
the vehicle. Any locally uniform surface works; absolute reflectance
is unknown, so only frame-to-frame changes are meaningful.

Vehicle probe: a pretrained MobileNet-SSD (OpenCV DNN) tracks a
leading vehicle. Per Weiss (2001), reflectance constancy of the
tracked patch means the median frame-to-frame log difference is a
direct illumination-ratio measurement, independent of the vehicle's
colour.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import IlluminationConfig

logger = logging.getLogger(__name__)

SSD_INPUT_SIZE = (300, 300)
SSD_SCALE = 0.007843
SSD_MEAN = 127.5
SSD_VEHICLE_CLASSES = (6, 7, 14)  # bus, car, motorbike in VOC ordering
MAX_DETECTIONS_KEPT = 10


class GroundProbe:
    """Tracks illumination change via a fixed ground patch."""

    def __init__(self, config: IlluminationConfig):
        """Initialise the probe.

        Args:
            config (IlluminationConfig): Validated illumination config.
        """
        self.config = config
        self._last_log_median: Optional[float] = None

    def reset(self):
        """Clear inter-frame state between videos."""
        self._last_log_median = None

    def measure(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Measure the log-illumination step since the last frame.

        Args:
            frame_bgr (np.ndarray): BGR frame (post-crop).

        Returns:
            Optional[float]: Delta log intensity, or None on the
                first frame of a video.

        Raises:
            ValueError: If the frame is empty.
        """
        if frame_bgr.size == 0:
            raise ValueError("frame must be non-empty")
        h, w = frame_bgr.shape[:2]
        fx1, fy1, fx2, fy2 = self.config.ground_patch_frac
        patch = frame_bgr[int(fy1 * h):int(fy2 * h), int(fx1 * w):int(fx2 * w)]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        log_median = float(np.median(np.log(gray.astype(np.float32) + 1.0)))
        delta = None
        if self._last_log_median is not None:
            delta = log_median - self._last_log_median
        self._last_log_median = log_median
        return delta


class VehicleProbe:
    """Tracks a leading vehicle as a constant-reflectance target."""

    def __init__(self, config: IlluminationConfig):
        """Load the detector if model files are configured and exist.

        Args:
            config (IlluminationConfig): Validated illumination config.
        """
        self.config = config
        self.net = None
        self._track_box: Optional[Tuple[int, int, int, int]] = None
        self._track_len = 0
        self._last_log_median: Optional[float] = None
        proto, weights = config.vehicle_model_prototxt, config.vehicle_model_weights
        if proto is None or weights is None:
            logger.info("Vehicle probe disabled: model paths not set")
            return
        if not Path(proto).exists() or not Path(weights).exists():
            logger.warning("Vehicle probe disabled: model files missing")
            return
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

    @property
    def available(self) -> bool:
        """Whether the detector loaded.

        Returns:
            bool: True if measurements are possible.
        """
        return self.net is not None

    def reset(self):
        """Clear tracking state between videos."""
        self._track_box = None
        self._track_len = 0
        self._last_log_median = None

    def measure(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Measure the illumination step from the tracked vehicle.

        Args:
            frame_bgr (np.ndarray): BGR frame (post-crop).

        Returns:
            Optional[float]: Delta log intensity of the tracked patch,
                or None when no stable track exists.
        """
        if not self.available:
            return None
        box = self._detect_and_associate(frame_bgr)
        if box is None:
            self.reset()
            return None
        x, y, w, h = box
        patch = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        if patch.size == 0:
            return None
        log_median = float(np.median(np.log(patch.astype(np.float32) + 1.0)))
        delta = None
        if (self._last_log_median is not None
                and self._track_len >= self.config.vehicle_min_track_frames):
            delta = log_median - self._last_log_median
        self._last_log_median = log_median
        return delta

    def _detect_and_associate(self, frame: np.ndarray):
        """Detect vehicles and associate to the running track by IoU.

        Args:
            frame (np.ndarray): BGR frame.

        Returns:
            Optional[Tuple[int, int, int, int]]: Tracked box, or None.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, SSD_INPUT_SIZE), SSD_SCALE, SSD_INPUT_SIZE, SSD_MEAN
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        # Sort by confidence explicitly; SSD output order is usually
        # confidence-sorted but that is not contractual.
        confs = detections[0, 0, :, 2]
        order = np.argsort(-confs)[:MAX_DETECTIONS_KEPT]
        boxes: List[Tuple[int, int, int, int]] = []
        for i in order:
            conf = float(detections[0, 0, i, 2])
            cls = int(detections[0, 0, i, 1])
            if conf < self.config.vehicle_confidence or cls not in SSD_VEHICLE_CLASSES:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        if not boxes:
            return None
        if self._track_box is None:
            best = max(boxes, key=lambda b: b[2] * b[3])
            self._track_box, self._track_len = best, 1
            return best
        best, best_iou = None, 0.0
        for b in boxes:
            iou = self._iou(b, self._track_box)
            if iou > best_iou:
                best, best_iou = b, iou
        if best is None or best_iou < self.config.vehicle_iou_min:
            best = max(boxes, key=lambda b: b[2] * b[3])
            self._track_box, self._track_len = best, 1
            self._last_log_median = None
            return best
        self._track_box = best
        self._track_len += 1
        return best

    @staticmethod
    def _iou(a, b) -> float:
        """Intersection over union of two (x, y, w, h) boxes.

        Args:
            a: First box.
            b: Second box.

        Returns:
            float: IoU in [0, 1].
        """
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0
