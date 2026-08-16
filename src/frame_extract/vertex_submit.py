"""Submit this repo as a Vertex AI Custom Job.

Mirrors the deployment pattern of the GCP_Cloud_Send desktop app (used for
the cctv_zarr model) without depending on that repo: the repo is packaged
as a tarball, staged to ``gs://<bucket>/vertex-jobs/<job>/source/``, and a
container command fetches + extracts it into /workspace, installs
``requirements.txt``, and runs ``scripts/vertex_entrypoint.sh``.

Credentials come from Application Default Credentials
(``gcloud auth application-default login`` or a service-account key via
``GOOGLE_APPLICATION_CREDENTIALS``).

Requires the ``cloud`` extra: ``pip install -e ".[cloud]"``.
"""

import logging
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Never ship these into the job tarball.
EXCLUDE_NAMES = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".DS_Store",
    "test_data", "extracted_frames", "mock_dashcam_out", "vertex_run_out",
    "input", "forecast_cache.json",
}

SOURCE_BLOB = "vertex-jobs/{job}/source/source.tar.gz"
REQUIREMENTS_BLOB = "vertex-jobs/{job}/source/requirements.txt"
CONSOLE_URL = ("https://console.cloud.google.com/vertex-ai/locations/"
               "{region}/training/{job_id}?project={project}")


@dataclass(frozen=True)
class CloudJobRequest:
    """Everything needed to submit one Custom Job."""

    project: str
    region: str
    staging_bucket: str
    image_uri: str
    machine_type: str = "n1-standard-4"
    job_name: str = "dashcam-poc"
    videos_uri: Optional[str] = None   # gs:// prefix of footage; None -> mock smoke
    gpx_uri: Optional[str] = None      # gs:// GPX track


def _require_sdk():
    """Import the GCP SDKs lazily with a clear error.

    Returns:
        tuple: (aiplatform, storage) modules.

    Raises:
        ImportError: If the ``cloud`` extra is not installed.
    """
    try:
        from google.cloud import aiplatform, storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-aiplatform / google-cloud-storage are not "
            "installed. Install the cloud extra: pip install -e \".[cloud]\""
        ) from e
    return aiplatform, storage


def _package_repo(archive_path: Path) -> None:
    """Tar the repo root, excluding VCS/venv/output clutter.

    Args:
        archive_path (Path): Destination .tar.gz path.
    """
    def keep(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        parts = Path(info.name).parts
        return None if any(p in EXCLUDE_NAMES for p in parts) else info

    with tarfile.open(archive_path, "w:gz") as tar:
        for entry in sorted(REPO_ROOT.iterdir()):
            if entry.name in EXCLUDE_NAMES:
                continue
            tar.add(entry, arcname=entry.name, filter=keep)


def submit(req: CloudJobRequest) -> dict:
    """Package, stage, and submit the Custom Job.

    Args:
        req (CloudJobRequest): Validated submission parameters.

    Returns:
        dict: job display name, resource name, and console URL.
    """
    if not (REPO_ROOT / "scripts" / "vertex_entrypoint.sh").exists():
        raise FileNotFoundError(
            "Repo layout not found next to the installed package; cloud "
            "submission requires running from a checkout of "
            "Cloud_Dashcam_POC (pip install -e)."
        )

    aiplatform, storage = _require_sdk()

    aiplatform.init(
        project=req.project,
        location=req.region,
        staging_bucket=f"gs://{req.staging_bucket}",
    )

    client = storage.Client(project=req.project)
    bucket = client.bucket(req.staging_bucket)

    archive = Path(tempfile.gettempdir()) / f"dashcam_src_{os.getpid()}.tar.gz"
    _package_repo(archive)
    source_blob = SOURCE_BLOB.format(job=req.job_name)
    try:
        bucket.blob(source_blob).upload_from_filename(str(archive))
    finally:
        archive.unlink(missing_ok=True)
    logger.info("Staged source to gs://%s/%s", req.staging_bucket, source_blob)

    code_uri = f"gs://{req.staging_bucket}/{source_blob}"
    command = (
        "mkdir -p /workspace && "
        f"gsutil cp {code_uri} /tmp/source.tar.gz && "
        "tar -xzf /tmp/source.tar.gz -C /workspace && cd /workspace && "
        "pip install -r requirements.txt && "
        "bash scripts/vertex_entrypoint.sh"
    )

    env = []
    if req.videos_uri:
        env.append({"name": "DASHCAM_VIDEOS_URI", "value": req.videos_uri})
    if req.gpx_uri:
        env.append({"name": "DASHCAM_GPX_URI", "value": req.gpx_uri})

    worker_pool_specs = [{
        "machine_spec": {"machine_type": req.machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": req.image_uri,
            "command": ["/bin/sh", "-c"],
            "args": [command],
            "env": env,
        },
    }]

    job = aiplatform.CustomJob(
        display_name=req.job_name,
        worker_pool_specs=worker_pool_specs,
        staging_bucket=f"gs://{req.staging_bucket}",
        base_output_dir=f"gs://{req.staging_bucket}/vertex-jobs/{req.job_name}",
    )
    job.submit()

    job_id = job.resource_name.split("/")[-1]
    return {
        "job_name": req.job_name,
        "resource_name": job.resource_name,
        "console_url": CONSOLE_URL.format(
            region=req.region, job_id=job_id, project=req.project,
        ),
        "outputs_uri": (f"gs://{req.staging_bucket}/vertex-jobs/"
                        f"{req.job_name}/"),
    }
