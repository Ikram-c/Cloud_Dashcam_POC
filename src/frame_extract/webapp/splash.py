"""Splash launcher: pick Local or Cloud with a toggle, then run.

A small standalone FastAPI app served before the main control panel:

- **Local** — starts the existing five-screen web control panel
  (``server.create_app``) in a background uvicorn thread on the
  ``ui.host``/``ui.port`` from config.yaml, and hands the browser off
  to it.
- **Cloud** — submits this repo as a Vertex AI Custom Job via
  :mod:`frame_extract.vertex_submit` (ADC credentials), using the same
  stage-tarball-and-run pattern GCP_Cloud_Send uses for the cctv model.

Run: ``frame-extract-splash`` (default http://127.0.0.1:8320).
"""

import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..config import Settings

logger = logging.getLogger(__name__)

SPLASH_PATH = Path(__file__).parent / "splash.html"
DEFAULT_SPLASH_PORT = 8320

MACHINE_TYPES = [
    "n1-standard-4", "n1-standard-8", "n1-standard-16",
    "n2-standard-4", "n2-standard-8",
    "e2-standard-4", "e2-standard-8",
]


class CloudSubmitRequest(BaseModel):
    """Cloud form payload from the browser."""

    project: str
    region: str = "us-central1"
    staging_bucket: str
    image_uri: str
    machine_type: str = "n1-standard-4"
    job_name: str = "dashcam-poc"
    videos_uri: Optional[str] = None
    gpx_uri: Optional[str] = None


class _CloudState:
    """Latest cloud submission outcome, guarded by one lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = "idle"        # idle | submitting | submitted | failed
        self._detail: dict = {}
        self._error: Optional[str] = None

    def try_start(self) -> bool:
        with self._lock:
            if self._state == "submitting":
                return False
            self._state = "submitting"
            self._detail = {}
            self._error = None
            return True

    def finish(self, detail: dict):
        with self._lock:
            self._state = "submitted"
            self._detail = dict(detail)

    def fail(self, message: str):
        with self._lock:
            self._state = "failed"
            self._error = message

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "detail": dict(self._detail),
                "error": self._error,
            }


class _LocalPanel:
    """Owns the background uvicorn thread for the main control panel."""

    def __init__(self, config_path: Path):
        self._lock = threading.Lock()
        self._config_path = config_path
        self._thread: Optional[threading.Thread] = None
        self._url: Optional[str] = None

    def start(self) -> str:
        """Start (or return the already-running) local panel.

        Returns:
            str: The panel URL to redirect the browser to.
        """
        import uvicorn

        from .server import create_app

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._url
            host, port = "127.0.0.1", 8321
            try:
                ui = Settings.load(self._config_path).ui
                host, port = ui.host, ui.port
            except (FileNotFoundError, KeyError, ValueError) as e:
                logger.warning("ui config unreadable (%s); using %s:%d",
                               e, host, port)
            server = uvicorn.Server(uvicorn.Config(
                create_app(self._config_path),
                host=host, port=port, log_level="info",
            ))
            self._thread = threading.Thread(target=server.run, daemon=True)
            self._thread.start()
            self._url = f"http://{host}:{port}/"
            return self._url


def _submit_worker(req: CloudSubmitRequest, state: _CloudState):
    """Background thread: run the Vertex submission."""
    from ..vertex_submit import CloudJobRequest, submit

    try:
        detail = submit(CloudJobRequest(
            project=req.project.strip(),
            region=req.region.strip(),
            staging_bucket=req.staging_bucket.strip().removeprefix("gs://"),
            image_uri=req.image_uri.strip(),
            machine_type=req.machine_type,
            job_name=req.job_name.strip() or "dashcam-poc",
            videos_uri=(req.videos_uri or "").strip() or None,
            gpx_uri=(req.gpx_uri or "").strip() or None,
        ))
        state.finish(detail)
    except Exception as e:  # noqa: BLE001 - thread boundary
        logger.exception("Cloud submission failed")
        state.fail(str(e))


def create_app(config_path: Path) -> FastAPI:
    """Build the splash app.

    Args:
        config_path (Path): Path to config.yaml (forwarded to the local
            panel when Local mode is chosen).

    Returns:
        FastAPI: The configured application.
    """
    app = FastAPI(title="Cloud Optimised DashCam — Launcher")
    cloud = _CloudState()
    panel = _LocalPanel(config_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return SPLASH_PATH.read_text(encoding="utf-8")

    @app.get("/api/options")
    def options() -> dict:
        return {"machine_types": MACHINE_TYPES}

    @app.post("/api/local/start")
    def local_start() -> dict:
        try:
            return {"url": panel.start()}
        except Exception as e:  # noqa: BLE001 - surface to browser
            logger.exception("Local panel failed to start")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cloud/submit")
    def cloud_submit(req: CloudSubmitRequest) -> dict:
        for field in ("project", "staging_bucket", "image_uri"):
            if not getattr(req, field).strip():
                raise HTTPException(status_code=400,
                                    detail=f"{field} is required")
        if not cloud.try_start():
            raise HTTPException(status_code=409,
                                detail="a submission is already in flight")
        threading.Thread(
            target=_submit_worker, args=(req, cloud), daemon=True,
        ).start()
        return {"submitting": True}

    @app.get("/api/cloud/status")
    def cloud_status() -> dict:
        return cloud.snapshot()

    return app


def main():
    """Entry point: serve the splash screen."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Local/Cloud launcher splash screen")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_SPLASH_PORT)
    args = parser.parse_args()
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
