#!/usr/bin/env python3
"""Generate synthetic test fixtures: videos, GPX telemetry, edge cases.

Fixtures produced under ./test_data:
  - sharp moving scene (should mostly pass the gates)
  - heavily blurred scene (blur gate)
  - static scene with a translating shadow (duplicate gate + Weiss)
  - near-black night scene (exposure gate)
  - a non-conforming filename (metadata fallback path)
  - a GPX telemetry track covering the fixture timestamps, moving
    through the vodafone urban_4g demo zone (coverage chunking)
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

FPS = 30.0
DURATION_S = 6
WIDTH, HEIGHT = 640, 480
TRACK_START = datetime(2024, 3, 15, 8, 0, tzinfo=timezone.utc)
TRACK_END = datetime(2024, 3, 16, 3, 0, tzinfo=timezone.utc)
TRACK_STEP_MIN = 10
LON_START, LON_END = -1.60, -1.48
LAT_START, LAT_END = 53.795, 53.815


def moving_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    """Render a sharp scene with strong translating texture.

    Args:
        t (float): Time in seconds; drives the translation.
        rng (np.random.Generator): Noise source.

    Returns:
        np.ndarray: BGR frame.
    """
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    shift = 80.0 * t
    pattern = (
        90.0 + 60.0 * np.sin(0.08 * (x - shift))
        + 30.0 * np.sin(0.15 * (y + 0.5 * shift))
        + rng.normal(0.0, 5.0, x.shape)
    )
    frame = np.clip(pattern, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def blurred_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    """Render the moving scene, then destroy sharpness.

    Args:
        t (float): Time in seconds.
        rng (np.random.Generator): Noise source.

    Returns:
        np.ndarray: Heavily Gaussian-blurred BGR frame.
    """
    return cv2.GaussianBlur(moving_frame(t, rng), (51, 51), 15)


def static_shadow_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    """Render a static textured scene with a translating shadow.

    Constant reflectance, varying illumination: the exact Weiss (2001)
    setting. The shadow sweeps left to right over the clip.

    Args:
        t (float): Time in seconds; drives the shadow position.
        rng (np.random.Generator): Noise source (fixed-seed texture).

    Returns:
        np.ndarray: BGR frame.
    """
    texture_rng = np.random.default_rng(7)
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    reflectance = (
        120.0 + 50.0 * np.sin(0.05 * x) * np.cos(0.07 * y)
        + texture_rng.normal(0.0, 8.0, x.shape)
    )
    shadow_left = (t / DURATION_S) * WIDTH - 120.0
    shadow = np.ones_like(x)
    in_shadow = (x >= shadow_left) & (x <= shadow_left + 240.0)
    shadow[in_shadow] = 0.45
    frame = np.clip(reflectance * shadow + rng.normal(0.0, 2.0, x.shape), 0, 255)
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def night_frame(t: float, rng: np.random.Generator) -> np.ndarray:
    """Render a near-black frame below all value thresholds.

    Args:
        t (float): Time in seconds (unused).
        rng (np.random.Generator): Noise source.

    Returns:
        np.ndarray: BGR frame.
    """
    return rng.integers(0, 10, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def write_video(path: Path, frame_fn) -> None:
    """Write a synthetic video using the given per-frame renderer.

    Args:
        path (Path): Output video path.
        frame_fn: Callable (t, rng) -> frame.
    """
    rng = np.random.default_rng(42)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for i in range(int(FPS * DURATION_S)):
        writer.write(frame_fn(i / FPS, rng))
    writer.release()
    print(f"Wrote {path}")


def write_gpx_track(path: Path) -> None:
    """Write a GPX track sweeping west-to-east across the demo zones.

    Fixes every TRACK_STEP_MIN minutes from TRACK_START to TRACK_END,
    linearly interpolating position so the route crosses the vodafone
    urban_4g and rural_edge demo bboxes in config.yaml. Plain string
    output; no gpxpy dependency for fixture generation.

    Args:
        path (Path): Output GPX path.
    """
    total = (TRACK_END - TRACK_START).total_seconds()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="make_test_data">',
        "  <trk><trkseg>",
    ]
    t = TRACK_START
    while t <= TRACK_END:
        f = (t - TRACK_START).total_seconds() / total
        lon = LON_START + f * (LON_END - LON_START)
        lat = LAT_START + f * (LAT_END - LAT_START)
        stamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(
            f'    <trkpt lat="{lat:.6f}" lon="{lon:.6f}">'
            f"<time>{stamp}</time></trkpt>"
        )
        t += timedelta(minutes=TRACK_STEP_MIN)
    lines += ["  </trkseg></trk>", "</gpx>", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main():
    """Generate the full fixture set under ./test_data."""
    root = Path("test_data")
    videos = root / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    write_video(videos / "CAR01_20240315_083000_CAM01.mp4", moving_frame)
    write_video(videos / "CAR01_20240315_103000_CAM01.mp4", blurred_frame)
    write_video(videos / "CAR01_20240315_120000_CAM01.mp4", static_shadow_frame)
    write_video(videos / "CAR01_20240316_020000_CAM01.mp4", night_frame)
    write_video(videos / "not_a_valid_name.mp4", moving_frame)

    write_gpx_track(root / "track.gpx")

    print("\nRun with:")
    print("  frame-extract --config config.yaml "
          "--videos test_data/videos --output extracted_frames "
          "--gpx test_data/track.gpx")


if __name__ == "__main__":
    main()
