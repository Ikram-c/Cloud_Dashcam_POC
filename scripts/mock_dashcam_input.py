#!/usr/bin/env python3
"""Headless mock-input stage for Cloud_Dashcam_POC.

The dashcam analog of cctv_zarr's ``scripts/mock_lens_input.py``: generate
synthetic dashcam footage + a GPX track, then run the full extraction
pipeline (quality gates, illumination with geometry-only priors, Weiss
reflectance, coverage chunking) end-to-end — fully offline, no credentials,
no network, no display.

Used three ways:
  - CI smoke stage (see .github/workflows/ci.yml)
  - Vertex AI Custom Job smoke run via GCP_Cloud_Send
    (see scripts/vertex_entrypoint.sh and DEPLOY.md)
  - local sanity check: ``python scripts/mock_dashcam_input.py``

No install required: ``src/`` is put on sys.path directly, mirroring how
GCP_Cloud_Send extracts the repo tarball into /workspace and runs a command.

Usage:
    python scripts/mock_dashcam_input.py --scene all --output mock_dashcam_out
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for entry in (str(REPO_ROOT / "src"), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import make_test_data  # noqa: E402  (repo scripts/, added to sys.path above)

# Scene name -> (fixture stem, renderer). Stems follow the
# VEHICLE_YYYYMMDD_HHMMSS_CAMERA pattern and fall inside the fixture
# GPX track's time window, so illumination and coverage both resolve.
# (The extension is chosen by the codec fallback below.)
SCENES = {
    "sharp": ("CAR01_20240315_083000_CAM01", make_test_data.moving_frame),
    "blur": ("CAR01_20240315_103000_CAM01", make_test_data.blurred_frame),
    "static": ("CAR01_20240315_120000_CAM01",
               make_test_data.static_shadow_frame),
    "night": ("CAR01_20240316_020000_CAM01", make_test_data.night_frame),
}

# Tried in order. cv2.VideoWriter fails *silently* when a codec is
# unavailable (common with source-built OpenCV lacking FFmpeg encoders),
# so every write is verified by reading the file back.
CODEC_CANDIDATES = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("MJPG", ".avi"),
    ("XVID", ".avi"),
)


def _readable(path: Path) -> bool:
    """Check a video opens and yields at least one frame."""
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


def write_video_verified(dest_stem: Path, frame_fn) -> Path:
    """Write one synthetic video, falling back through codecs.

    Args:
        dest_stem (Path): Destination path without extension.
        frame_fn: Callable (t, rng) -> BGR frame (from make_test_data).

    Returns:
        Path: The verified on-disk video path.

    Raises:
        RuntimeError: If no available codec produced a readable file.
    """
    fps, duration = make_test_data.FPS, make_test_data.DURATION_S
    size = (make_test_data.WIDTH, make_test_data.HEIGHT)
    for fourcc, ext in CODEC_CANDIDATES:
        path = dest_stem.with_suffix(ext)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size,
        )
        if not writer.isOpened():
            writer.release()
            path.unlink(missing_ok=True)
            continue
        rng = np.random.default_rng(42)
        for i in range(int(fps * duration)):
            writer.write(frame_fn(i / fps, rng))
        writer.release()
        if path.exists() and _readable(path):
            print(f"Wrote {path} ({fourcc})")
            return path
        path.unlink(missing_ok=True)
    raise RuntimeError(
        "cv2.VideoWriter could not encode with any of "
        f"{[c for c, _ in CODEC_CANDIDATES]}; this OpenCV build has no "
        "usable video encoder. Install the opencv-python (or "
        "opencv-python-headless) wheel rather than a source build, or use "
        "a Python version with prebuilt OpenCV wheels (e.g. 3.11-3.13)."
    )


def build_fixtures(fixture_dir: Path, scenes) -> Path:
    """Write synthetic videos and the GPX track.

    Args:
        fixture_dir (Path): Directory for videos/ and track.gpx.
        scenes: Iterable of scene names from SCENES.

    Returns:
        Path: Path to the written GPX track.
    """
    videos = fixture_dir / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        stem, renderer = SCENES[scene]
        write_video_verified(videos / stem, renderer)
    gpx = fixture_dir / "track.gpx"
    make_test_data.write_gpx_track(gpx)
    return gpx


def derive_offline_config(base_config: Path, out_dir: Path, fixture_dir: Path,
                          gpx: Path) -> Path:
    """Derive a fully offline config from the repo's config.yaml.

    Forces: local GPX telemetry, geometry-only illumination (no forecast
    HTTP calls), coverage chunking on (the fixture track crosses the demo
    vodafone zones), Weiss reflectance on, sequential execution.

    Args:
        base_config (Path): The repo's config.yaml.
        out_dir (Path): Run output directory.
        fixture_dir (Path): Directory containing videos/.
        gpx (Path): Fixture GPX track.

    Returns:
        Path: Path to the derived YAML config.
    """
    with base_config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw["telemetry"].update(
        gpx_path=str(gpx), gcs_bucket=None, gcs_blob=None, use_mock_gcs=True
    )
    raw["illumination"].update(
        forecast_provider="none",
        cache_path=str(out_dir / "forecast_cache.json"),
    )
    raw["coverage"]["enabled"] = True
    raw["intrinsic"]["enabled"] = True
    raw["runtime"].update(
        video_directory=str(fixture_dir / "videos"),
        output_directory=str(out_dir / "extracted"),
        num_workers=0,
    )

    derived = out_dir / "mock_config.yaml"
    with derived.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return derived


def main() -> int:
    """Generate fixtures, run the pipeline, and verify the artifacts."""
    # Surface pipeline logger output (per-video errors, gate warnings);
    # without this, extraction failures are silently swallowed.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Offline mock-input smoke run (dashcam analog of "
                    "cctv_zarr's mock_lens_input.py)")
    parser.add_argument("--scene", default="all",
                        choices=["all", *SCENES],
                        help="Which synthetic scene(s) to generate")
    parser.add_argument("--output", type=Path,
                        default=Path("mock_dashcam_out"),
                        help="Directory for fixtures, config, and results")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "config.yaml",
                        help="Base config to derive the offline config from")
    parser.add_argument("--keep-fixtures", action="store_true",
                        help="Keep the synthetic videos after the run")
    args = parser.parse_args()

    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = out_dir / "fixtures"

    print(f"[mock-input] generating scenes: {', '.join(scenes)}")
    gpx = build_fixtures(fixture_dir, scenes)

    derived = derive_offline_config(args.config, out_dir, fixture_dir, gpx)
    print(f"[mock-input] derived offline config: {derived}")

    from frame_extract.config import Settings
    from frame_extract.pipeline import ExtractionPipeline

    settings = Settings.load(derived)
    summaries = ExtractionPipeline(settings).run()

    for s in summaries:
        print(f"[mock-input] {Path(s.video_path).name}: "
              f"sampled={s.frames_sampled} kept={s.frames_kept} "
              f"blur={s.frames_rejected_blur} "
              f"exposure={s.frames_rejected_exposure} "
              f"duplicate={s.frames_rejected_duplicate} "
              f"reflectance={s.reflectance_frames}")
    if not summaries:
        print("[mock-input] FAIL: no videos were processed (scanner found "
              "none, or every extraction errored — see log lines above)")
        return 1

    extracted = out_dir / "extracted"
    manifest = extracted / settings.output.manifest_name
    chunks = extracted / settings.coverage.chunk_manifest_name
    if not manifest.exists():
        print(f"[mock-input] FAIL: no frames kept by any gate; manifest "
              f"not produced: {manifest} (see per-video counts above)")
        return 1
    print(f"[mock-input] OK: {manifest}")
    print(f"[mock-input] coverage chunks: "
          f"{chunks if chunks.exists() else 'not produced'}")

    if not args.keep_fixtures:
        shutil.rmtree(fixture_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
