#!/usr/bin/env python3
"""Fit sky-model constants to real footage.

Workflow: run the extractor over one day of footage with
``quality.enabled: false`` and ``illumination.enabled: false`` (so
dark and bright frames stay in the manifest), then run this script
against the manifest. It resolves per-frame timestamps, interpolates
position from the shared GPX telemetry (or the fixed location), fits
the intensity model by least squares, and prints YAML-ready constants.

Model fitted:
    log(grey+1) = night + (clear - night) * min(1, m + sin(elev))
                  - atten * cloud
"""

import argparse
import sys
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from frame_extract.config import Settings
from frame_extract.gnss import GnssTrack
from frame_extract.sky_model import SkyModel, solar_position
from frame_extract.telemetry import TelemetryProvider
from frame_extract.video_scanner import VideoScanner

MIN_SAMPLES = 50


def main():
    """Fit and report calibrated sky-model constants."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plot", action="store_true",
                        help="Show a fit plot (requires matplotlib)")
    args = parser.parse_args()

    settings = Settings.load(args.config)
    zone = ZoneInfo(settings.illumination.timezone)
    scanner = VideoScanner(settings.runtime)
    provider = TelemetryProvider(settings)
    nodes = None
    if provider.configured:
        nodes = provider.nodes(args.manifest.parent)
        print(f"Telemetry: {len(nodes)} GPX fixes")
    else:
        print("No telemetry configured; using the fixed location")
    track = GnssTrack(settings.illumination, nodes)
    sky = SkyModel(settings.illumination)

    df = pd.read_csv(args.manifest)
    rows = []
    for _, r in df.iterrows():
        meta = scanner.parse_filename(Path(str(r["video_path"])))
        if meta is None:
            continue
        dt = meta["file_datetime"].replace(tzinfo=zone).astimezone(timezone.utc)
        dt = dt + timedelta(seconds=float(r["timestamp_sec"]))
        fix = track.position(dt)
        if fix is None:
            continue
        elev, _ = solar_position(dt, fix.lat, fix.lon)
        cloud, _ = sky._forecast(dt, fix.lat, fix.lon)
        rows.append({
            "log_mean": float(np.log(float(r["mean_intensity"]) + 1.0)),
            "elev": elev,
            "cloud": cloud if cloud is not None else 0.0,
        })
    if len(rows) < MIN_SAMPLES:
        raise SystemExit(
            f"Only {len(rows)} usable samples; need a full day of footage"
        )
    data = pd.DataFrame(rows)

    m = settings.illumination.min_sun_factor
    sun = np.minimum(1.0, m + np.sin(np.radians(np.maximum(data.elev, 0.0))))
    design = np.column_stack([
        np.ones(len(data)), sun.to_numpy(), -data.cloud.to_numpy()
    ])
    coef, _, _, _ = np.linalg.lstsq(design, data.log_mean.to_numpy(), rcond=None)
    night, span, atten = coef
    clear = night + span
    rmse = float(np.sqrt(np.mean((design @ coef - data.log_mean.to_numpy()) ** 2)))

    print(f"Samples: {len(data)}   fit RMSE: {rmse:.3f} log units "
          f"({np.exp(rmse):.2f}x intensity)")
    print("\nSuggested config.yaml values (illumination section):")
    print(f"  night_log_intensity: {night:.2f}")
    print(f"  clear_sky_log_intensity: {clear:.2f}")
    print(f"  cloud_attenuation_log: {max(atten, 0.0):.2f}")
    if atten < 0:
        print("  # WARNING: negative cloud coefficient - the forecast is not")
        print("  # explaining brightness variation. Check timezone alignment,")
        print("  # or the footage may be single-condition (all clear/overcast).")
    if rmse > 0.7:
        print("  # WARNING: high RMSE - auto-exposure may be compensating")
        print("  # heavily. Consider lowering fusion_gain toward 0.1 and")
        print("  # widening exposure_tolerance_stops.")
    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter(data.elev, data.log_mean, s=4, alpha=0.4, label="frames")
        order = np.argsort(data.elev.to_numpy())
        plt.plot(data.elev.to_numpy()[order], (design @ coef)[order], "r-",
                 label="fit")
        plt.xlabel("sun elevation (deg)")
        plt.ylabel("log(mean grey + 1)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()