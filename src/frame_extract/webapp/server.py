"""Web control panel: pick footage, tune gates, run the pipeline.

FastAPI app serving a single-page panel built to platform interface
guidelines: system typography at a 17px base with fluid headings,
light and dark modes with at least 4.5:1 text contrast, safe-area
insets, a fixed 49px tab bar with three destinations that becomes a
260-320px sidebar on wide screens, 44px minimum touch targets, pill
toggles for the quality gates, contained scroll views, spinners
deferred one second, per-frame batch progress, and implicit
auto-save of panel state at most every thirty seconds.

Videos are selected by clicking: a native file picker (with a folder
variant and drag-and-drop) copies footage to this machine in bounded
chunked uploads, and a server-side folder browser covers footage
already here. A run processes the explicit selection, a whole
folder, or the built-in local demo.
"""

import argparse
import base64
import binascii
import json
import logging
import shutil
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import yaml

from ..config import Settings
from ..exceptions import VideoOpenError
from ..models import ProgressEvent
from ..pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)

MAX_FOLDERS_LISTED = 200
MAX_VIDEOS_LISTED = 500
MAX_SELECTED_VIDEOS = 500
MAX_ACTIVE_UPLOADS = 8
MAX_UPLOAD_PART_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_NAME_COLLISIONS = 100
PREFS_FILENAME = "panel_prefs.json"
SELECTED_DIRNAME = "_selected_run"
UPLOAD_PREFIX = ".upload_"


class BrowseRequest(BaseModel):
    """Folder listing request."""

    path: Optional[str] = None


class UploadRequest(BaseModel):
    """One part of a chunked video upload from the browser."""

    upload_id: str = Field(min_length=1, max_length=64,
                           pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    seq: int = Field(ge=0)
    last: bool = False
    data: str = ""


class ExtractRequest(BaseModel):
    """One extraction launch request from the panel.

    ``gates`` maps gate names (blur, exposure, duplicate,
    illumination, weiss, coverage) to booleans; a True is ANDed with
    the YAML, so the browser can disable what config enables but
    never the reverse. ``selected`` names explicit video files;
    ``folder`` processes a whole folder; ``videos``/``output`` are
    directory overrides.
    """

    gates: Dict[str, bool] = Field(default_factory=dict)
    network: Optional[str] = None
    gpx_source: Optional[str] = None
    sampling: Optional[str] = None
    videos: Optional[str] = None
    output: Optional[str] = None
    gpx_path: Optional[str] = None
    selected: List[str] = Field(default_factory=list)
    folder: Optional[str] = None


class PrefsRequest(BaseModel):
    """Implicit auto-save payload for panel state."""

    prefs: Dict[str, object] = Field(default_factory=dict)


def _apply_request(settings: Settings, req: ExtractRequest) -> Settings:
    """Fold a browser request into an immutable settings copy.

    Args:
        settings (Settings): Settings loaded from config.yaml.
        req (ExtractRequest): The browser's request.

    Returns:
        Settings: A new settings object with the request applied.

    Raises:
        ValueError: On an unknown network or sampling choice.
    """
    g = req.gates
    quality = settings.quality
    if not g.get("blur", True):
        quality = replace(quality, blur_threshold=0.0)
    if not g.get("exposure", True):
        quality = replace(
            quality, min_mean_intensity=-1.0, max_mean_intensity=256.0,
        )
    if not g.get("duplicate", True):
        quality = replace(quality, duplicate_threshold=0.0)

    illumination = replace(
        settings.illumination,
        enabled=settings.illumination.enabled and g.get("illumination", True),
    )
    intrinsic = replace(
        settings.intrinsic,
        enabled=settings.intrinsic.enabled and g.get("weiss", True),
    )

    coverage = settings.coverage
    if req.network is not None:
        if req.network not in coverage.networks:
            raise ValueError(
                f"unknown network '{req.network}'; "
                f"choose from {sorted(coverage.networks)}"
            )
        coverage = replace(coverage, network=req.network)
    coverage = replace(
        coverage, enabled=coverage.enabled and g.get("coverage", True),
    )

    telemetry = settings.telemetry
    if req.gpx_source == "mock":
        telemetry = replace(telemetry, use_mock_gcs=True, gpx_path=None)
    elif req.gpx_source == "gcs":
        telemetry = replace(telemetry, use_mock_gcs=False, gpx_path=None)
    if req.gpx_path:
        telemetry = replace(telemetry, gpx_path=req.gpx_path)

    sampling = settings.sampling
    if req.sampling:
        kind, _, value = req.sampling.partition("_")
        if kind == "frames":
            sampling = replace(
                sampling, mode="frames", every_n_frames=int(value),
            )
        elif kind == "seconds":
            sampling = replace(
                sampling, mode="seconds", every_n_seconds=float(value),
            )
        else:
            raise ValueError(f"unknown sampling choice '{req.sampling}'")

    runtime = settings.runtime
    if req.videos:
        runtime = replace(runtime, video_directory=req.videos)
    if req.output:
        runtime = replace(runtime, output_directory=req.output)

    return replace(
        settings, quality=quality, intrinsic=intrinsic,
        illumination=illumination, coverage=coverage,
        telemetry=telemetry, sampling=sampling, runtime=runtime,
    )


def _read_zone_counts(output_root: Path, manifest_name: str) -> dict:
    """Read per-zone slice counts from the coverage chunk manifest.

    Args:
        output_root (Path): Pipeline output directory.
        manifest_name (str): The chunk manifest filename.

    Returns:
        dict: zone -> number of slice ranges; {} when absent.
    """
    path = Path(output_root) / manifest_name
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Unreadable chunk manifest %s: %s", path, e)
        return {}
    zones = payload.get("zones") or {}
    return {zone: len(entries or []) for zone, entries in zones.items()}


def _summary_dict(summary) -> dict:
    """Serialise one VideoSummary for the panel.

    Args:
        summary: The VideoSummary.

    Returns:
        dict: JSON-safe fields.
    """
    raw = asdict(summary)
    return {
        "video": Path(raw["video_path"]).name,
        "frames_sampled": raw["frames_sampled"],
        "frames_kept": raw["frames_kept"],
        "rejected_blur": raw["frames_rejected_blur"],
        "rejected_exposure": raw["frames_rejected_exposure"],
        "rejected_duplicate": raw["frames_rejected_duplicate"],
        "reflectance_frames": raw["reflectance_frames"],
    }


class JobState:
    """Thread-safe single-run state behind the status API."""

    def __init__(self):
        """Initialise the idle state."""
        self._lock = threading.Lock()
        self.state = "idle"
        self.progress = 0.0
        self.network: Optional[str] = None
        self.zones: dict = {}
        self.error: Optional[str] = None
        self.stats: dict = {}
        self.current_video = ""
        self.video_index = 0
        self.video_count = 0
        self.summaries: List[dict] = []

    def try_start(self, network: Optional[str]) -> bool:
        """Claim the single run slot.

        Args:
            network (Optional[str]): The selected carrier.

        Returns:
            bool: False when a run is already active.
        """
        with self._lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.progress = 0.0
            self.network = network
            self.zones = {}
            self.error = None
            self.stats = {}
            self.current_video = ""
            self.video_index = 0
            self.video_count = 0
            self.summaries = []
            return True

    def update(self, event: ProgressEvent):
        """Fold one progress observation into the snapshot.

        Args:
            event (ProgressEvent): Emitted by the pipeline.
        """
        with self._lock:
            frac = (
                event.frames_done / event.frames_total
                if event.frames_total > 0 else 1.0
            )
            count = max(event.video_count, 1)
            self.progress = min(1.0, (event.video_index + frac) / count)
            self.current_video = Path(event.video_path).name
            self.video_index = event.video_index
            self.video_count = event.video_count
            self.stats = dict(event.stats)

    def finish(self, summaries: List[dict], zones: dict,
               network: Optional[str]):
        """Mark the run done.

        Args:
            summaries (List[dict]): Serialised per-video summaries.
            zones (dict): Per-zone slice counts.
            network (Optional[str]): The carrier the run used.
        """
        with self._lock:
            self.state = "done"
            self.progress = 1.0
            self.zones = zones
            self.network = network
            self.summaries = summaries
            self.current_video = ""

    def fail(self, error: str):
        """Mark the run failed.

        Args:
            error (str): Human-readable failure description.
        """
        with self._lock:
            self.state = "failed"
            self.error = error

    def snapshot(self) -> dict:
        """Atomically copy the state for the status endpoint.

        Returns:
            dict: state, progress, network, zones, stats, summaries.
        """
        with self._lock:
            return {
                "state": self.state,
                "progress": self.progress,
                "network": self.network,
                "zones": dict(self.zones),
                "error": self.error,
                "stats": dict(self.stats),
                "current_video": self.current_video,
                "video_index": self.video_index,
                "video_count": self.video_count,
                "summaries": list(self.summaries),
            }


def _stage_selection(settings: Settings, selected: List[str]) -> Settings:
    """Stage explicit video files into a private run directory.

    Args:
        settings (Settings): Applied settings.
        selected (List[str]): Absolute video file paths.

    Returns:
        Settings: Settings whose video_directory is the staged run.

    Raises:
        ValueError: When a selected file does not exist.
    """
    run_dir = Path(settings.runtime.output_directory) / SELECTED_DIRNAME
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for position, raw in enumerate(selected[:MAX_SELECTED_VIDEOS]):
        source = Path(raw).expanduser()
        if not source.is_file():
            raise ValueError(f"That video was not found: {source}")
        link = run_dir / f"{position:03d}_{source.name}"
        try:
            link.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, link)
    return replace(
        settings,
        runtime=replace(settings.runtime, video_directory=str(run_dir)),
    )


def _run_job(settings: Settings, job: JobState):
    """Daemon-thread body: run the pipeline and settle the job state.

    Args:
        settings (Settings): Fully applied settings.
        job (JobState): The shared job state.
    """
    try:
        pipeline = ExtractionPipeline(settings)
        summaries = pipeline.run(progress=job.update)
        zones = _read_zone_counts(
            Path(settings.runtime.output_directory),
            settings.coverage.chunk_manifest_name,
        )
        job.finish(
            [_summary_dict(s) for s in summaries],
            zones, settings.coverage.network,
        )
    except Exception as e:
        logger.exception("Extraction run failed")
        job.fail(str(e))


def create_app(config_path: Path) -> FastAPI:
    """Build the control-panel app bound to one config file.

    Args:
        config_path (Path): Path to config.yaml.

    Returns:
        FastAPI: The application.
    """
    app = FastAPI(title="Cloud Optimised DashCam")
    job = JobState()
    uploads: Dict[str, dict] = {}
    uploads_lock = threading.Lock()
    config_path = Path(config_path)

    def _settings_or_400() -> Settings:
        """Load settings or raise a 400.

        Returns:
            Settings: Validated settings.

        Raises:
            HTTPException: When the config cannot be loaded.
        """
        try:
            return Settings.load(config_path)
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the panel."""
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/status")
    def status() -> dict:
        """Report the run state."""
        return job.snapshot()

    @app.get("/api/networks")
    def networks() -> dict:
        """List selectable mobile networks."""
        settings = _settings_or_400()
        return {
            "networks": sorted(settings.coverage.networks),
            "default": settings.coverage.network,
        }

    @app.post("/api/browse")
    def browse(req: BrowseRequest) -> dict:
        """List one folder's subfolders and selectable videos."""
        settings = _settings_or_400()
        base = (
            Path(req.path).expanduser()
            if req.path else Path(settings.runtime.video_directory)
        )
        try:
            base = base.resolve()
        except OSError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not base.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"That folder was not found: {base}",
            )
        output_root = Path(settings.runtime.output_directory)
        try:
            children = sorted(base.iterdir())
        except OSError as e:
            raise HTTPException(status_code=400, detail=str(e))
        folders = [
            p.name for p in children
            if p.is_dir() and not p.name.startswith(".")
        ][:MAX_FOLDERS_LISTED]
        allowed = {e.lower() for e in settings.runtime.video_extensions}
        videos = []
        for p in children:
            if len(videos) >= MAX_VIDEOS_LISTED:
                break
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            videos.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "processed": (output_root / p.stem).is_dir(),
            })
        return {
            "path": str(base),
            "parent": str(base.parent),
            "folders": folders,
            "videos": videos,
        }

    def _validate_upload_name(settings: Settings, name: str) -> str:
        """Validate a picked file name and its extension.

        Args:
            settings (Settings): Root configuration.
            name (str): The browser-supplied file name.

        Returns:
            str: The safe file name.

        Raises:
            HTTPException: On unsafe names or non-video extensions.
        """
        if Path(name).name != name or name.startswith("."):
            raise HTTPException(
                status_code=400,
                detail="That file name is not allowed.",
            )
        allowed = {e.lower() for e in settings.runtime.video_extensions}
        if Path(name).suffix.lower() not in allowed:
            raise HTTPException(
                status_code=400,
                detail="That file does not look like a video.",
            )
        return name

    def _final_upload_path(folder: Path, name: str) -> Path:
        """Pick a non-colliding destination for an uploaded video.

        Args:
            folder (Path): The video directory.
            name (str): The safe file name.

        Returns:
            Path: A free destination path.

        Raises:
            HTTPException: When too many name collisions exist.
        """
        candidate = folder / name
        stem, suffix = candidate.stem, candidate.suffix
        for attempt in range(1, MAX_NAME_COLLISIONS + 1):
            if not candidate.exists():
                return candidate
            candidate = folder / f"{stem}_{attempt}{suffix}"
        raise HTTPException(
            status_code=400,
            detail="Too many files with that name already exist.",
        )

    @app.post("/api/upload")
    def upload(req: UploadRequest) -> dict:
        """Receive one part of a clicked-or-dropped video upload."""
        settings = _settings_or_400()
        name = _validate_upload_name(settings, req.name)
        try:
            payload = base64.b64decode(req.data, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(
                status_code=400,
                detail="That upload was not readable.",
            )
        if len(payload) > MAX_UPLOAD_PART_BYTES:
            raise HTTPException(
                status_code=400,
                detail="That upload part is too large.",
            )
        folder = Path(settings.runtime.video_directory)
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / (UPLOAD_PREFIX + req.upload_id)
        with uploads_lock:
            entry = uploads.get(req.upload_id)
            if entry is None:
                if req.seq != 0:
                    raise HTTPException(
                        status_code=400,
                        detail="That upload was interrupted. "
                               "Please try again.",
                    )
                if len(uploads) >= MAX_ACTIVE_UPLOADS:
                    raise HTTPException(
                        status_code=409,
                        detail="Too many uploads at once. "
                               "Please wait a moment.",
                    )
                entry = {"name": name, "next_seq": 0, "bytes": 0}
                uploads[req.upload_id] = entry
                temp.write_bytes(b"")
            if req.seq != entry["next_seq"]:
                uploads.pop(req.upload_id, None)
                temp.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="That upload was interrupted. "
                           "Please try again.",
                )
            if entry["bytes"] + len(payload) > MAX_UPLOAD_TOTAL_BYTES:
                uploads.pop(req.upload_id, None)
                temp.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="That video is too large to copy.",
                )
            with temp.open("ab") as handle:
                handle.write(payload)
            entry["next_seq"] += 1
            entry["bytes"] += len(payload)
            if not req.last:
                return {"done": False, "received": entry["bytes"]}
            uploads.pop(req.upload_id, None)
            final = _final_upload_path(folder, entry["name"])
            temp.rename(final)
        return {"done": True, "path": str(final)}

    @app.post("/api/extract")
    def extract(req: ExtractRequest) -> dict:
        """Launch one pipeline run in a daemon thread."""
        try:
            settings = _apply_request(Settings.load(config_path), req)
            if req.folder:
                folder = Path(req.folder).expanduser()
                if not folder.is_dir():
                    raise ValueError(
                        f"That folder was not found: {folder}"
                    )
                settings = replace(
                    settings,
                    runtime=replace(
                        settings.runtime, video_directory=str(folder),
                    ),
                )
            if req.selected:
                settings = _stage_selection(settings, req.selected)
        except (FileNotFoundError, KeyError, ValueError,
                VideoOpenError) as e:
            job.fail(str(e))
            raise HTTPException(status_code=400, detail=str(e))
        if not job.try_start(settings.coverage.network):
            raise HTTPException(
                status_code=409,
                detail="A run is already in progress. "
                       "Please wait for it to finish.",
            )
        threading.Thread(
            target=_run_job, args=(settings, job), daemon=True,
        ).start()
        return {"started": True, "network": settings.coverage.network}

    @app.get("/api/prefs")
    def read_prefs() -> dict:
        """Return the auto-saved panel state."""
        try:
            settings = Settings.load(config_path)
        except (FileNotFoundError, KeyError, ValueError):
            return {"prefs": {}}
        path = Path(settings.runtime.output_directory) / PREFS_FILENAME
        if not path.is_file():
            return {"prefs": {}}
        try:
            return {"prefs": json.loads(path.read_text())}
        except (ValueError, OSError):
            return {"prefs": {}}

    @app.post("/api/prefs")
    def write_prefs(req: PrefsRequest) -> dict:
        """Persist the auto-saved panel state."""
        settings = _settings_or_400()
        root = Path(settings.runtime.output_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / PREFS_FILENAME).write_text(json.dumps(req.prefs))
        return {"saved": True}

    return app


def main():
    """Entry point for the frame-extract-ui console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="DashCam web panel")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    settings = Settings.load(args.config)
    import uvicorn
    uvicorn.run(
        create_app(args.config),
        host=settings.ui.host, port=settings.ui.port,
    )


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Cloud Optimised DashCam</title>
<style>
:root {
  color-scheme: light dark;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --quiet: #e1e0d9;
  --danger: #d03b3b; --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --quiet: #2c2c2a;
    --danger: #e66767; --good: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
  font-size: 17px; line-height: 1.45;
  background: var(--page); color: var(--ink-1);
  padding: env(safe-area-inset-top) env(safe-area-inset-right)
           0 env(safe-area-inset-left);
}
.shell { display: block; min-height: 100%; }
.sidebar { display: none; }
.content {
  padding: 16px;
  padding-bottom: calc(49px + 16px + env(safe-area-inset-bottom));
  max-width: 760px; margin: 0 auto;
}
.masthead { display: flex; align-items: center; gap: 12px;
  margin-bottom: 16px; }
.appicon {
  width: 44px; height: 44px; border-radius: 22.5%;
  background: linear-gradient(135deg, var(--series-1), #104281);
  flex: none;
}
h1 { font-size: clamp(22px, 2.2vw + 17px, 28px); }
h2 { font-size: clamp(19px, 1vw + 17px, 22px); margin-bottom: 12px; }
.sub { color: var(--ink-2); font-size: 17px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
button {
  font: inherit; font-size: 17px;
  min-height: 44px; min-width: 44px; padding: 0 18px;
  border-radius: 10px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--ink-1); cursor: pointer;
}
button.primary {
  background: var(--series-1); border-color: var(--series-1);
  color: #ffffff; font-weight: 600;
}
button:disabled { opacity: 0.5; cursor: default; }
input[type=text], select {
  font: inherit; font-size: 17px; appearance: none;
  width: 100%; min-height: 44px; padding: 8px 12px;
  border-radius: 10px; border: 1px solid var(--baseline);
  background: var(--surface-1); color: var(--ink-1);
}
label.field {
  display: block; color: var(--ink-2); font-size: 17px;
  margin: 12px 0 4px;
}
.row { display: flex; gap: 12px; align-items: center;
  flex-wrap: wrap; margin-top: 12px; }
.toggle { display: inline-flex; align-items: center; gap: 10px;
  min-height: 44px; cursor: pointer; }
.toggle input { position: absolute; opacity: 0; }
.knob {
  width: 51px; height: 31px; border-radius: 999px;
  background: var(--baseline); position: relative;
  transition: background .2s; flex: none;
}
.knob::after {
  content: ""; position: absolute; top: 2px; left: 2px;
  width: 27px; height: 27px; border-radius: 999px;
  background: #ffffff; transition: left .2s;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.toggle input:checked + .knob { background: var(--series-1); }
.toggle input:checked + .knob::after { left: 22px; }
.gates { display: grid; grid-template-columns: 1fr 1fr;
  gap: 4px 16px; }
.list { overscroll-behavior: contain; touch-action: pan-y;
  max-height: 380px; overflow-y: auto; }
.cell {
  display: flex; align-items: center; gap: 12px;
  min-height: 44px; padding: 6px 0; cursor: pointer;
}
.cell + .cell { border-top: 1px solid var(--grid); }
.cell .grow { flex: 1; min-width: 0; }
.cell .title { font-size: 17px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.cell .meta { color: var(--ink-muted); font-size: 15px; }
.cell .badge { color: var(--good); font-size: 15px; flex: none; }
.cell .chev { color: var(--ink-muted); flex: none; }
.check {
  width: 28px; height: 28px; border-radius: 999px;
  border: 2px solid var(--baseline); flex: none;
  display: flex; align-items: center; justify-content: center;
  color: transparent; font-size: 16px; font-weight: 700;
}
.cell.selected .check {
  background: var(--series-1); border-color: var(--series-1);
  color: #ffffff;
}
.dot { width: 10px; height: 10px; border-radius: 999px;
  background: var(--quiet); flex: none; }
.dot.on { background: var(--series-1); }
.crumb { color: var(--ink-muted); font-size: 15px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  margin-bottom: 4px; }
.drop {
  border: 2px dashed var(--baseline); border-radius: 12px;
  padding: 20px 16px; text-align: center; color: var(--ink-2);
  margin-top: 12px;
}
.drop.hover { border-color: var(--series-1); color: var(--ink-1); }
.hiddeninput { display: none; }
.progress { height: 8px; border-radius: 999px; background: var(--grid);
  overflow: hidden; flex: 1; min-width: 120px; }
.progress > div { height: 100%; width: 0%;
  background: var(--series-1); border-radius: 999px;
  transition: width .3s; }
.tab-bar {
  position: fixed; bottom: 0; left: 0; right: 0;
  height: calc(49px + env(safe-area-inset-bottom));
  padding-bottom: env(safe-area-inset-bottom);
  display: flex; background: var(--surface-1);
  border-top: 1px solid var(--border); z-index: 40;
}
.tab-bar button {
  flex: 1; border: 0; border-radius: 0; background: none;
  color: var(--ink-muted); font-size: 15px; min-height: 49px;
}
.tab-bar button.active { color: var(--series-1); font-weight: 600; }
.view { display: none; }
.view.active { display: block; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  background: var(--page); border: 1px solid var(--border);
  border-radius: 999px; padding: 4px 12px; font-size: 15px;
  color: var(--ink-2);
}
.chip b { color: var(--good); }
.spinner {
  width: 28px; height: 28px; border-radius: 999px;
  border: 3px solid var(--grid); border-top-color: var(--series-1);
  animation: spin 1s linear infinite; display: none;
}
@keyframes spin { to { transform: rotate(360deg); } }
.error { color: var(--danger); margin-top: 8px; min-height: 22px; }
.statgrid { display: flex; flex-direction: column; gap: 4px;
  margin-top: 8px; }
.statgrid .line { display: flex; justify-content: space-between;
  gap: 16px; color: var(--ink-2); }
.statgrid .line b { color: var(--ink-1);
  font-variant-numeric: tabular-nums; }
@media (min-width: 900px) {
  .shell {
    display: grid;
    grid-template-columns: minmax(260px, 320px) 1fr;
    min-height: 100vh;
  }
  .sidebar {
    display: block; background: var(--surface-1);
    border-right: 1px solid var(--border); padding: 16px;
  }
  .tab-bar { display: none; }
  .content { padding-bottom: 16px; }
  .sidebar nav button {
    display: block; width: 100%; text-align: left; border: 0;
    background: none; color: var(--ink-1); margin-bottom: 2px;
  }
  .sidebar nav button.active {
    background: var(--page); color: var(--series-1); font-weight: 600;
    border-radius: 10px;
  }
  .only-narrow { display: none; }
}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="masthead">
      <div class="appicon"></div>
      <div><h1>Cloud Optimised DashCam</h1>
      <div class="sub">Quality-checked footage, organised</div></div>
    </div>
    <nav>
      <button data-tab="extract" class="active">Add footage</button>
      <button data-tab="options">Options</button>
      <button data-tab="results">Results</button>
    </nav>
  </aside>
  <main class="content">
    <div class="masthead only-narrow">
      <div class="appicon"></div>
      <div><h1>Cloud Optimised DashCam</h1>
      <div class="sub">Quality-checked footage, organised</div></div>
    </div>

    <section class="view active" id="view-extract">
      <div class="card">
        <h2>Select Video File</h2>
        <div class="drop" id="dropzone">
          <div>Drop videos here, or pick them with a click</div>
          <div class="row" style="justify-content: center;">
            <button class="primary" id="btn-pick-files">
              Choose files</button>
            <button id="btn-pick-folder">Choose a folder</button>
          </div>
          <input type="file" id="file-input" class="hiddeninput"
                 multiple accept="video/*">
          <input type="file" id="folder-input" class="hiddeninput"
                 webkitdirectory>
        </div>
        <div class="row" id="upload-row" hidden>
          <div class="progress"><div id="upload-fill"></div></div>
          <span class="sub" id="upload-label"></span>
        </div>
        <label class="field" for="folder-path">
          Or browse this computer</label>
        <input type="text" id="folder-path" autocomplete="off"
               placeholder="videos">
        <div class="row">
          <button id="btn-open">Open folder</button>
          <button id="btn-select-all">Select all</button>
        </div>
        <div class="crumb" id="crumb"></div>
        <div class="list" id="browser"></div>
        <div class="row">
          <button class="primary" id="btn-run-selected" disabled>
            Process selected</button>
          <button id="btn-run-folder">Process whole folder</button>
          <button id="btn-demo">Run Local Demo</button>
          <div class="spinner" id="extract-spinner"></div>
        </div>
        <div class="error" id="extract-error"></div>
      </div>
    </section>

    <section class="view" id="view-options">
      <div class="card">
        <h2>Quality checks</h2>
        <div class="gates">
          <label class="toggle"><input type="checkbox" id="g-blur"
            checked><span class="knob"></span>
            <span>Skip blurry frames</span></label>
          <label class="toggle"><input type="checkbox" id="g-exposure"
            checked><span class="knob"></span>
            <span>Skip too dark or bright</span></label>
          <label class="toggle"><input type="checkbox" id="g-duplicate"
            checked><span class="knob"></span>
            <span>Skip repeated frames</span></label>
          <label class="toggle"><input type="checkbox"
            id="g-illumination" checked><span class="knob"></span>
            <span>Daylight awareness</span></label>
          <label class="toggle"><input type="checkbox" id="g-weiss"
            checked><span class="knob"></span>
            <span>Shadow-free snapshots</span></label>
          <label class="toggle"><input type="checkbox" id="g-coverage"
            checked><span class="knob"></span>
            <span>Group by phone signal</span></label>
        </div>
      </div>
      <div class="card">
        <h2>Route and network</h2>
        <label class="field" for="network">Mobile network</label>
        <select id="network"></select>
        <label class="field" for="gpx-source">Route recording</label>
        <select id="gpx-source">
          <option value="local">Use my saved route file</option>
          <option value="gcs">Fetch from the cloud</option>
          <option value="mock">Demo route</option>
        </select>
        <label class="field" for="sampling">How often to keep frames
        </label>
        <select id="sampling">
          <option value="seconds_1.0">Every second</option>
          <option value="seconds_0.5">Twice a second</option>
          <option value="frames_30">Every 30 frames</option>
          <option value="frames_10">Every 10 frames</option>
        </select>
      </div>
    </section>

    <section class="view" id="view-results">
      <div class="card" id="running-card" hidden>
        <h2>Working&hellip;</h2>
        <div class="row">
          <div class="progress"><div id="run-fill"></div></div>
          <span class="sub" id="run-count"></span>
        </div>
        <div class="sub" id="run-current"></div>
        <div class="sub" id="run-stats"></div>
      </div>
      <div class="card" id="done-card" hidden>
        <h2>Processing Complete</h2>
        <div class="sub" id="done-line"></div>
        <div class="chips" id="zone-chips"></div>
        <div class="list" id="summary-list"></div>
      </div>
      <div class="card" id="idle-card">
        <h2>No results yet</h2>
        <div class="sub">Add footage and start a run to see results
        here.</div>
      </div>
    </section>
  </main>
</div>

<nav class="tab-bar">
  <button data-tab="extract" class="active">Add footage</button>
  <button data-tab="options">Options</button>
  <button data-tab="results">Results</button>
</nav>

<script>
"use strict";
const AUTO_SAVE_MS = 30000;
const SPINNER_DELAY_MS = 1000;
const POLL_MS = 400;
const selected = new Set();
let currentFolder = "";
let pollTimer = null;
let dirty = false;

function haptic(ms) {
  if (navigator.vibrate) { navigator.vibrate(ms); }
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  };
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || ("HTTP " + res.status));
  }
  return data;
}

function switchTab(name) {
  document.querySelectorAll("[data-tab]").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.id === "view-" + name));
  markDirty();
}
document.querySelectorAll("[data-tab]").forEach(b =>
  b.addEventListener("click", () => {
    haptic(10); switchTab(b.dataset.tab);
  }));

function activeTab() {
  const btn = document.querySelector(".tab-bar button.active");
  return btn ? btn.dataset.tab : "extract";
}

async function loadNetworks() {
  try {
    const data = await api("/api/networks");
    const sel = document.getElementById("network");
    sel.innerHTML = "";
    for (const n of data.networks) {
      const opt = document.createElement("option");
      opt.value = n; opt.textContent = n;
      if (n === data.default) { opt.selected = true; }
      sel.appendChild(opt);
    }
  } catch (e) {}
}

function updateSelectedButton() {
  const btn = document.getElementById("btn-run-selected");
  btn.disabled = selected.size === 0;
  btn.textContent = selected.size
    ? "Process selected (" + selected.size + ")" : "Process selected";
}

function videoCell(v) {
  const cell = document.createElement("div");
  cell.className = "cell" + (selected.has(v.path) ? " selected" : "");
  cell.innerHTML =
    '<span class="check">\\u2713</span><div class="grow">' +
    '<div class="title">' + v.name + '</div><div class="meta">' +
    v.size_mb + " MB</div></div>" +
    (v.processed ? '<span class="badge">Processed</span>' : "");
  cell.addEventListener("click", () => {
    haptic(10);
    if (selected.has(v.path)) { selected.delete(v.path); }
    else { selected.add(v.path); }
    cell.classList.toggle("selected", selected.has(v.path));
    updateSelectedButton();
  });
  return cell;
}

function folderCell(name, target) {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.innerHTML =
    '<div class="grow"><div class="title">' + name +
    '</div></div><span class="chev">\\u203a</span>';
  cell.addEventListener("click", () => { haptic(10); browse(target); });
  return cell;
}

async function browse(path) {
  const errorEl = document.getElementById("extract-error");
  errorEl.textContent = "";
  try {
    const data = await api("/api/browse", path ? {path: path} : {});
    currentFolder = data.path;
    selected.clear();
    updateSelectedButton();
    document.getElementById("folder-path").value = data.path;
    document.getElementById("crumb").textContent = data.path;
    const list = document.getElementById("browser");
    list.textContent = "";
    list.appendChild(folderCell("..", data.parent));
    for (const name of data.folders) {
      list.appendChild(folderCell(name, data.path + "/" + name));
    }
    for (const v of data.videos) { list.appendChild(videoCell(v)); }
    if (!data.folders.length && !data.videos.length) {
      list.innerHTML += '<div class="cell"><div class="grow">' +
        '<div class="meta">Empty folder</div></div></div>';
    }
    markDirty();
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-open").addEventListener("click", () =>
  browse(document.getElementById("folder-path").value.trim() || null));
document.getElementById("btn-select-all")
  .addEventListener("click", () => {
    haptic(10);
    const all = document.querySelectorAll("#browser .cell");
    let any = false;
    all.forEach(cell => {
      if (cell.querySelector(".check") &&
          !cell.classList.contains("selected")) {
        any = true;
        cell.click();
      }
    });
    if (!any) {
      all.forEach(cell => {
        if (cell.querySelector(".check") &&
            cell.classList.contains("selected")) {
          cell.click();
        }
      });
    }
  });

function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () =>
      reject(new Error("Could not read that file"));
    reader.readAsDataURL(blob);
  });
}

function isVideoName(name) {
  return /\\.(mp4|avi|mkv|mov)$/i.test(name);
}

async function uploadPicked(fileList) {
  const files = Array.from(fileList)
    .filter(f => isVideoName(f.name)).slice(0, 50);
  const errorEl = document.getElementById("extract-error");
  errorEl.textContent = "";
  if (!files.length) {
    errorEl.textContent = "No videos were picked.";
    return;
  }
  const row = document.getElementById("upload-row");
  const fill = document.getElementById("upload-fill");
  const label = document.getElementById("upload-label");
  row.hidden = false;
  const partSize = 8 * 1024 * 1024;
  const totalBytes = files.reduce((n, f) => n + Math.max(f.size, 1), 0);
  let sentBytes = 0;
  const paths = [];
  try {
    for (let f = 0; f < files.length; f++) {
      const file = files[f];
      label.textContent = "Copying " + (f + 1) + " of " +
        files.length + ": " + file.name;
      const id = "u" + Date.now().toString(36) +
        Math.random().toString(36).slice(2, 10);
      const parts = Math.max(1, Math.ceil(file.size / partSize));
      for (let i = 0; i < parts; i++) {
        const blob = file.slice(i * partSize, (i + 1) * partSize);
        const data = await blobToB64(blob);
        const res = await api("/api/upload", {
          upload_id: id, name: file.name, seq: i,
          last: i === parts - 1, data: data,
        });
        sentBytes += Math.max(blob.size, 1);
        fill.style.width =
          Math.round(100 * sentBytes / totalBytes) + "%";
        if (res.path) { paths.push(res.path); }
      }
    }
    label.textContent = "Copied " + paths.length + " video(s)";
    haptic(10);
    await launch({selected: paths});
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  } finally {
    setTimeout(() => { row.hidden = true; }, 1500);
  }
}

document.getElementById("btn-pick-files")
  .addEventListener("click", () => {
    haptic(10);
    document.getElementById("file-input").click();
  });
document.getElementById("btn-pick-folder")
  .addEventListener("click", () => {
    haptic(10);
    document.getElementById("folder-input").click();
  });
document.getElementById("file-input")
  .addEventListener("change", ev => {
    uploadPicked(ev.target.files);
    ev.target.value = "";
  });
document.getElementById("folder-input")
  .addEventListener("change", ev => {
    uploadPicked(ev.target.files);
    ev.target.value = "";
  });
const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("dragover", ev => {
  ev.preventDefault();
  dropzone.classList.add("hover");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("hover"));
dropzone.addEventListener("drop", ev => {
  ev.preventDefault();
  dropzone.classList.remove("hover");
  haptic(10);
  if (ev.dataTransfer && ev.dataTransfer.files) {
    uploadPicked(ev.dataTransfer.files);
  }
});

function gates() {
  return {
    blur: document.getElementById("g-blur").checked,
    exposure: document.getElementById("g-exposure").checked,
    duplicate: document.getElementById("g-duplicate").checked,
    illumination: document.getElementById("g-illumination").checked,
    weiss: document.getElementById("g-weiss").checked,
    coverage: document.getElementById("g-coverage").checked,
  };
}

function baseBody() {
  return {
    gates: gates(),
    network: document.getElementById("network").value || null,
    gpx_source: document.getElementById("gpx-source").value,
    sampling: document.getElementById("sampling").value,
  };
}

async function launch(extra) {
  const errorEl = document.getElementById("extract-error");
  errorEl.textContent = "";
  const body = Object.assign(baseBody(), extra || {});
  try {
    await api("/api/extract", body);
    haptic(10);
    switchTab("results");
    document.getElementById("idle-card").hidden = true;
    document.getElementById("done-card").hidden = true;
    document.getElementById("running-card").hidden = false;
    poll();
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-run-selected")
  .addEventListener("click", () =>
    launch({selected: Array.from(selected)}));
document.getElementById("btn-run-folder")
  .addEventListener("click", () =>
    launch({folder: currentFolder ||
      document.getElementById("folder-path").value.trim() || null}));
document.getElementById("btn-demo").addEventListener("click", () =>
  launch({gates: {}, gpx_source: "mock"}));

function renderRunning(s) {
  document.getElementById("run-fill").style.width =
    Math.round(s.progress * 100) + "%";
  document.getElementById("run-count").textContent =
    s.video_count
      ? "video " + (s.video_index + 1) + " of " + s.video_count : "";
  document.getElementById("run-current").textContent =
    s.current_video ? "Working on " + s.current_video : "";
  const st = s.stats || {};
  document.getElementById("run-stats").textContent =
    st.sampled != null
      ? "checked " + st.sampled + " frames - skipped " +
        ((st.blur || 0) + (st.underexposed || 0) +
         (st.overexposed || 0) + (st.duplicate || 0)) +
        " - snapshots " + (st.reflectance || 0)
      : "";
}

function renderDone(s) {
  document.getElementById("running-card").hidden = true;
  document.getElementById("idle-card").hidden = true;
  const card = document.getElementById("done-card");
  card.hidden = false;
  const kept = s.summaries.reduce((n, r) => n + r.frames_kept, 0);
  document.getElementById("done-line").textContent =
    s.summaries.length + " video(s) processed - " + kept +
    " frames kept" + (s.network ? " - network " + s.network : "");
  const chips = document.getElementById("zone-chips");
  chips.textContent = "";
  const zones = s.zones || {};
  for (const zone of Object.keys(zones)) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = zone + ": <b>" + zones[zone] + "</b>";
    chips.appendChild(chip);
  }
  if (!Object.keys(zones).length) {
    chips.innerHTML =
      '<span class="chip">no signal grouping this run</span>';
  }
  const list = document.getElementById("summary-list");
  list.textContent = "";
  for (const r of s.summaries) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.innerHTML =
      '<span class="dot on"></span><div class="grow">' +
      '<div class="title">' + r.video + '</div>' +
      '<div class="meta">kept ' + r.frames_kept + " of " +
      r.frames_sampled + " - blurry " + r.rejected_blur +
      " - exposure " + r.rejected_exposure +
      " - repeats " + r.rejected_duplicate +
      " - snapshots " + r.reflectance_frames + "</div></div>";
    list.appendChild(cell);
  }
}

function poll() {
  clearTimeout(pollTimer);
  const spinner = document.getElementById("extract-spinner");
  const timer = setTimeout(() => { spinner.style.display = "block"; },
                           SPINNER_DELAY_MS);
  const tick = async () => {
    if (document.hidden) {
      pollTimer = setTimeout(tick, POLL_MS);
      return;
    }
    try {
      const s = await api("/api/status");
      if (s.state === "running") {
        renderRunning(s);
        pollTimer = setTimeout(tick, POLL_MS);
        return;
      }
      clearTimeout(timer);
      spinner.style.display = "none";
      if (s.state === "done") {
        haptic(10);
        renderDone(s);
        browse(currentFolder || null);
      } else if (s.state === "failed") {
        haptic(30);
        document.getElementById("running-card").hidden = true;
        document.getElementById("extract-error").textContent =
          s.error || "The run failed.";
        switchTab("extract");
      }
    } catch (e) {
      clearTimeout(timer);
      spinner.style.display = "none";
      document.getElementById("extract-error").textContent = e.message;
    }
  };
  pollTimer = setTimeout(tick, POLL_MS);
}

function markDirty() { dirty = true; }
["folder-path"].forEach(id => document.getElementById(id)
  .addEventListener("input", markDirty));
["g-blur", "g-exposure", "g-duplicate", "g-illumination",
 "g-weiss", "g-coverage"].forEach(id =>
  document.getElementById(id).addEventListener("change", () => {
    haptic(10); markDirty();
  }));
["network", "gpx-source", "sampling"].forEach(id =>
  document.getElementById(id).addEventListener("change", markDirty));

async function autoSave() {
  if (!dirty) { return; }
  dirty = false;
  const prefs = {
    tab: activeTab(),
    folder: currentFolder,
    gates: gates(),
    network: document.getElementById("network").value,
    gpx_source: document.getElementById("gpx-source").value,
    sampling: document.getElementById("sampling").value,
  };
  try { await api("/api/prefs", {prefs: prefs}); } catch (e) {}
}
setInterval(autoSave, AUTO_SAVE_MS);
window.addEventListener("pagehide", autoSave);

async function restore() {
  let folder = null;
  await loadNetworks();
  try {
    const data = await api("/api/prefs");
    const p = data.prefs || {};
    if (p.gates) {
      for (const key of Object.keys(p.gates)) {
        const el = document.getElementById("g-" + key);
        if (el) { el.checked = Boolean(p.gates[key]); }
      }
    }
    if (p.network) {
      document.getElementById("network").value = p.network;
    }
    if (p.gpx_source) {
      document.getElementById("gpx-source").value = p.gpx_source;
    }
    if (p.sampling) {
      document.getElementById("sampling").value = p.sampling;
    }
    if (p.folder) { folder = p.folder; }
    if (p.tab) { switchTab(p.tab); }
  } catch (e) {}
  browse(folder);
}
restore();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
