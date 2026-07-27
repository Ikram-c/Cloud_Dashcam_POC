#!/usr/bin/env python3
"""CLI entry point. All tunables live in config.yaml; the CLI carries
only paths and the GPX telemetry override."""

import argparse
import logging
from pathlib import Path

from .config import Settings
from .pipeline import ExtractionPipeline


def main():
    """Parse arguments, load settings, and run the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Quality-gated frame extraction")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--videos", type=Path, default=None, help="Video dir override")
    parser.add_argument("--output", type=Path, default=None, help="Output dir override")
    parser.add_argument("--gpx", type=Path, default=None,
                        help="GPX telemetry override (illumination + coverage)")
    args = parser.parse_args()
    settings = Settings.load(args.config).with_overrides(
        video_directory=args.videos,
        output_directory=args.output,
        gpx=args.gpx,
    )
    ExtractionPipeline(settings).run()


if __name__ == "__main__":
    main()
