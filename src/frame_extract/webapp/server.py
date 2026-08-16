"""FastAPI web control panel: launch runs, watch progress, review results.

Serves a dark five-screen flow (welcome -> demo -> configure ->
processing -> complete) plus a footage-review player that draws a
semi-transparent processing-timeline overlay (chunk creation, data
upstream windows) on top of a posted video, driven by the artifacts
of the last pipeline run.

Gate toggles neutralise thresholds via ``dataclasses.replace`` (a
disabled gate can never fire) and AND with the YAML: the browser can
disable what config enables, never the reverse. One job runs at a
time; concurrent launches receive 409.
"""

import csv
import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..config import Settings
from ..models import ProgressEvent, VideoSummary
from ..pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

INDEX_PATH = Path(__file__).parent / "index.html"

GATE_KEYS = ("blur", "exposure", "duplicate", "illumination", "weiss", "coverage")

# Neutral thresholds: values at which a gate can never fire.
NEUTRAL_BLUR_THRESHOLD = 0.0
NEUTRAL_MIN_INTENSITY = -1.0
NEUTRAL_MAX_INTENSITY = 256.0
NEUTRAL_DUPLICATE_THRESHOLD = 0.0

FALLBACK_FRAME_BYTES = 400_000
TIMELINE_STEP_SECONDS = 0.25


class ExtractRequest(BaseModel):
    """One extraction launch request from the browser."""

    videos: Optional[str] = None
    output: Optional[str] = None
    gpx_source: Optional[str] = None      # config | local | mock | none
    gpx_path: Optional[str] = None
    network: Optional[str] = None
    sampling: Optional[str] = None        # e.g. "seconds_1", "frames_30"
    gates: Dict[str, bool] = {}


def _apply_request(settings: Settings, req: ExtractRequest) -> Settings:
    """Fold a browser request into an immutable settings copy.

    Args:
        settings (Settings): Config-loaded baseline.
        req (ExtractRequest): Browser overrides and gate toggles.

    Returns:
        Settings: New settings with the request applied.

    Raises:
        ValueError: For an unknown network or malformed sampling.
    """
    out = settings.with_overrides(
        video_directory=Path(req.videos) if req.videos else None,
        output_directory=Path(req.output) if req.output else None,
    )
    if req.gpx_source == "mock":
        out = replace(out, telemetry=replace(
            out.telemetry,
            gpx_path=None,
            gcs_bucket=out.telemetry.gcs_bucket or "mock-bucket",
            gcs_blob=out.telemetry.gcs_blob or "telemetry/track.gpx",
            use_mock_gcs=True,
        ))
    elif req.gpx_source == "local" and req.gpx_path:
        out = replace(out, telemetry=replace(
            out.telemetry, gpx_path=req.gpx_path,
        ))
    elif req.gpx_source == "none":
        out = replace(out, telemetry=replace(
            out.telemetry, gpx_path=None, gcs_bucket=None, gcs_blob=None,
        ))
    if req.network is not None:
        if req.network not in out.coverage.networks:
            raise ValueError(
                f"unknown network '{req.network}'; "
                f"expected one of {sorted(out.coverage.networks)}"
            )
        out = replace(out, coverage=replace(out.coverage, network=req.network))
    if req.sampling is not None:
        out = replace(out, sampling=_parse_sampling(out, req.sampling))

    def enabled(key: str) -> bool:
        return req.gates.get(key, True)

    quality = out.quality
    if not enabled("blur"):
        quality = replace(quality, blur_threshold=NEUTRAL_BLUR_THRESHOLD)
    if not enabled("exposure"):
        quality = replace(
            quality,
            min_mean_intensity=NEUTRAL_MIN_INTENSITY,
            max_mean_intensity=NEUTRAL_MAX_INTENSITY,
        )
    if not enabled("duplicate"):
        quality = replace(quality, duplicate_threshold=NEUTRAL_DUPLICATE_THRESHOLD)
    return replace(
        out,
        quality=quality,
        illumination=replace(
            out.illumination,
            enabled=out.illumination.enabled and enabled("illumination"),
        ),
        intrinsic=replace(
            out.intrinsic,
            enabled=out.intrinsic.enabled and enabled("weiss"),
        ),
        coverage=replace(
            out.coverage,
            enabled=out.coverage.enabled and enabled("coverage"),
        ),
    )


def _parse_sampling(settings: Settings, choice: str):
    """Translate a browser sampling choice into a SamplingConfig.

    Args:
        settings (Settings): Baseline (for the untouched fields).
        choice (str): "<mode>_<value>", e.g. "frames_30", "seconds_2".

    Returns:
        SamplingConfig: The updated sampling section.

    Raises:
        ValueError: If the choice is malformed.
    """
    mode, _, value = choice.partition("_")
    if mode == "frames" and value.isdigit():
        return replace(
            settings.sampling, mode="frames", every_n_frames=int(value),
        )
    if mode == "seconds":
        try:
            return replace(
                settings.sampling, mode="seconds", every_n_seconds=float(value),
            )
        except ValueError as e:
            raise ValueError(f"bad sampling choice '{choice}'") from e
    raise ValueError(f"bad sampling choice '{choice}'")


def _read_zone_counts(output_root: Path, manifest_name: str) -> dict:
    """Read per-zone chunk counts from the coverage manifest.

    Args:
        output_root (Path): Pipeline output directory.
        manifest_name (str): Coverage manifest filename.

    Returns:
        dict: zone -> number of upload windows; {} when absent.
    """
    path = Path(output_root) / manifest_name
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}
    zones = payload.get("zones") or {}
    return {zone: len(ranges or []) for zone, ranges in zones.items()}


class JobState:
    """Single-job state machine guarded by one lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"
        self._progress = 0.0
        self._video: Optional[str] = None
        self._network: Optional[str] = None
        self._zones: dict = {}
        self._error: Optional[str] = None
        self._summaries: List[dict] = []

    def try_start(self, network: Optional[str]) -> bool:
        """Atomically claim the single job slot.

        Args:
            network (Optional[str]): Selected carrier, for the snapshot.

        Returns:
            bool: False when a job is already running.
        """
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._progress = 0.0
            self._video = None
            self._network = network
            self._zones = {}
            self._error = None
            self._summaries = []
            return True

    def update(self, event: ProgressEvent):
        """Fold one pipeline progress observation into the snapshot.

        Args:
            event (ProgressEvent): Per-frame or per-video observation.
        """
        with self._lock:
            videos = max(1, event.video_count)
            frames = max(1, event.frames_total)
            fraction = (event.video_index + event.frames_done / frames) / videos
            self._progress = min(1.0, max(self._progress, fraction))
            self._video = event.video_path.name

    def finish(self, summaries: List[VideoSummary], zones: dict,
               network: Optional[str]):
        """Record completion.

        Args:
            summaries (List[VideoSummary]): Per-video outcomes.
            zones (dict): zone -> chunk counts from coverage.
            network (Optional[str]): Carrier the run used.
        """
        with self._lock:
            self._state = "done"
            self._progress = 1.0
            self._network = network
            self._zones = dict(zones)
            self._summaries = [{
                "video": Path(s.video_path).name,
                "fps": s.fps,
                "total_frames": s.total_frames,
                "sampled": s.frames_sampled,
                "kept": s.frames_kept,
                "blur": s.frames_rejected_blur,
                "exposure": s.frames_rejected_exposure,
                "duplicate": s.frames_rejected_duplicate,
                "reflectance": s.reflectance_frames,
            } for s in summaries]

    def fail(self, message: str):
        """Record a failure.

        Args:
            message (str): Human-readable cause.
        """
        with self._lock:
            self._state = "failed"
            self._error = message

    def snapshot(self) -> dict:
        """Return an atomic copy of the visible state."""
        with self._lock:
            return {
                "state": self._state,
                "progress": self._progress,
                "video": self._video,
                "network": self._network,
                "zones": dict(self._zones),
                "error": self._error,
                "summaries": list(self._summaries),
            }


def _run_job(settings: Settings, job: JobState):
    """Worker-thread body: run the pipeline and settle the job state.

    Args:
        settings (Settings): Fully applied settings.
        job (JobState): Shared state to update.
    """
    try:
        summaries = ExtractionPipeline(settings).run(progress=job.update)
        zones = _read_zone_counts(
            Path(settings.runtime.output_directory),
            settings.coverage.chunk_manifest_name,
        )
        job.finish(summaries, zones, settings.coverage.network)
    except Exception as e:  # noqa: BLE001 - thread boundary
        logger.exception("Pipeline run failed")
        job.fail(str(e))


def _build_timeline(settings: Settings, video: Optional[str]) -> Optional[dict]:
    """Derive overlay timeline events from the last run's artifacts.

    Sources: ``frame_manifest.csv`` (per-frame capture times and real
    on-disk frame sizes), per-video ``kept_slices.yaml`` (chunk
    creation), ``coverage_chunks.yaml`` (upstream windows), and
    ``video_summaries.csv`` (fps, duration).

    Args:
        settings (Settings): Loaded settings (for paths and names).
        video (Optional[str]): Requested video stem; when absent or
            unmatched, the stem with the most kept frames is used.

    Returns:
        Optional[dict]: Timeline payload, or None when no manifest
            exists (no run yet).
    """
    output_root = Path(settings.runtime.output_directory)
    manifest = output_root / settings.output.manifest_name
    if not manifest.exists():
        return None
    by_stem: Dict[str, list] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["video_path"]).stem
            try:
                t = float(row["timestamp_sec"])
            except (KeyError, ValueError):
                continue
            by_stem.setdefault(stem, []).append((t, row.get("output_path", "")))
    if not by_stem:
        return None
    requested = Path(video).stem if video else None
    matched = requested in by_stem
    stem = requested if matched else max(by_stem, key=lambda s: len(by_stem[s]))

    fps, duration = _video_stats(output_root, stem, settings)

    series = []
    cumulative_bytes = 0
    sizes_seen = []
    for t, output_path in sorted(by_stem[stem]):
        size = None
        if output_path:
            p = Path(output_path)
            if p.exists():
                size = p.stat().st_size
        if size is None:
            size = int(sum(sizes_seen) / len(sizes_seen)) if sizes_seen \
                else FALLBACK_FRAME_BYTES
        else:
            sizes_seen.append(size)
        cumulative_bytes += size
        series.append({
            "t": round(t, 3),
            "frames": len(series) + 1,
            "bytes": cumulative_bytes,
        })

    chunks = []
    slices_path = output_root / stem / "kept_slices.yaml"
    if slices_path.exists():
        try:
            payload = yaml.safe_load(slices_path.read_text(encoding="utf-8")) or {}
            for item in payload.get("slices", []):
                chunks.append({
                    "start": item["start"] / fps,
                    "end": item["end"] / fps,
                    "t": item["end"] / fps,
                    "frames": item["end"] - item["start"],
                })
        except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning("Could not read %s: %s", slices_path, e)

    upstream = []
    network = settings.coverage.network
    coverage_path = output_root / settings.coverage.chunk_manifest_name
    if coverage_path.exists():
        try:
            payload = yaml.safe_load(coverage_path.read_text(encoding="utf-8")) or {}
            network = payload.get("network", network)
            for zone, ranges in (payload.get("zones") or {}).items():
                for item in ranges or []:
                    if Path(str(item.get("video", ""))).stem != stem:
                        continue
                    upstream.append({
                        "t0": item["start"] / fps,
                        "t1": item["end"] / fps,
                        "zone": zone,
                    })
        except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
            logger.warning("Could not read %s: %s", coverage_path, e)
    upstream.sort(key=lambda w: w["t0"])

    return {
        "video": stem,
        "matched": matched,
        "videos": sorted(by_stem),
        "fps": fps,
        "duration": duration,
        "network": network,
        "series": series,
        "chunks": chunks,
        "upstream": upstream,
        "step": TIMELINE_STEP_SECONDS,
    }


def _video_stats(output_root: Path, stem: str, settings: Settings):
    """Look up fps and duration for a video from the summaries CSV.

    Args:
        output_root (Path): Pipeline output directory.
        stem (str): Video stem.
        settings (Settings): For the default fps fallback.

    Returns:
        Tuple[float, float]: (fps, duration_seconds).
    """
    fps = settings.runtime.default_fps
    duration = 0.0
    summaries = output_root / "video_summaries.csv"
    if summaries.exists():
        with summaries.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if Path(row.get("video_path", "")).stem != stem:
                    continue
                try:
                    fps = float(row["fps"]) or fps
                    duration = int(row["total_frames"]) / fps
                except (KeyError, ValueError, ZeroDivisionError):
                    pass
                break
    return fps, duration


def create_app(config_path: Path) -> FastAPI:
    """Build the FastAPI app around one config file and one job slot.

    Args:
        config_path (Path): Path to config.yaml; loaded lazily so a
            missing file surfaces as a 400 on launch, not at startup.

    Returns:
        FastAPI: The configured application.
    """
    app = FastAPI(title="Cloud Optimised DashCam")
    job = JobState()
    cache: dict = {"settings": None}

    def load_settings() -> Settings:
        if cache["settings"] is None:
            cache["settings"] = Settings.load(Path(config_path))
        return cache["settings"]

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_PATH.read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict:
        return job.snapshot()

    @app.get("/api/networks")
    def networks() -> dict:
        settings = load_settings()
        return {
            "networks": sorted(settings.coverage.networks),
            "default": settings.coverage.network,
        }

    @app.post("/api/extract")
    def extract(req: ExtractRequest) -> dict:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            job.fail(f"config: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        try:
            applied = _apply_request(settings, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not job.try_start(applied.coverage.network):
            raise HTTPException(status_code=409, detail="a job is already running")
        cache["last_output"] = applied.runtime.output_directory
        threading.Thread(
            target=_run_job, args=(applied, job), daemon=True,
        ).start()
        return {"started": True, "network": applied.coverage.network}

    @app.get("/api/timeline")
    def timeline(video: Optional[str] = None) -> dict:
        try:
            settings = load_settings()
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        if cache.get("last_output"):
            settings = replace(settings, runtime=replace(
                settings.runtime, output_directory=cache["last_output"],
            ))
        payload = _build_timeline(settings, video)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail="no run artifacts found; run an extraction first",
            )
        return payload

    return app


def main():
    """Entry point: serve the panel on the configured host/port."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="Frame extraction web panel")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--host", default=None,
                        help="Override ui.host (e.g. 0.0.0.0 for containers)")
    parser.add_argument("--port", type=int, default=None,
                        help="Override ui.port (e.g. $PORT on Cloud Run)")
    args = parser.parse_args()
    host, port = "127.0.0.1", 8321
    try:
        ui = Settings.load(args.config).ui
        host, port = ui.host, ui.port
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning("Could not read ui config (%s); using %s:%d", e, host, port)
    if args.host is not None:
        host = args.host
    if args.port is not None:
        port = args.port
    uvicorn.run(create_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
