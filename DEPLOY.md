# Deploying Cloud_Dashcam_POC

Two deployment paths live in this repo:

1. **Demo web app on Cloud Run** — host the web control panel at a public
   HTTPS URL you can spin up for demos (this section).
2. **Vertex AI Custom Job via GCP_Cloud_Send** — batch extraction in the
   cloud (later sections).

## Demo web app on Cloud Run

The panel (welcome → demo → configure → processing → results) is
containerised by `Dockerfile.web` with demo fixtures pre-generated, so the
**Run Local Demo** button works entirely in-container — no data, no
telemetry setup, no credentials in the container.

One-time setup (pick your project/region once):

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com
gcloud artifacts repositories create dashcam \
    --repository-format=docker --location=us-central1
```

Build and deploy (repeat for each new version):

```bash
gcloud builds submit --config cloudbuild.web.yaml

gcloud run deploy dashcam-demo \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/dashcam/dashcam-web:latest \
    --region us-central1 \
    --memory 1Gi --cpu 1 \
    --no-cpu-throttling \
    --max-instances 1 \
    --allow-unauthenticated
```

The command prints the service URL (`https://dashcam-demo-….run.app`) —
open it, click **Run Local Demo**, and the full pipeline (gates,
illumination, coverage chunking, footage review overlay) runs live.

Why those flags:

- `--no-cpu-throttling` — extraction runs in a background thread while the
  browser polls `/api/status`; default Cloud Run throttles CPU between
  requests, which would stall the job.
- `--max-instances 1` — job state lives in the instance's memory; a second
  instance would answer polls with no knowledge of the running job.
- `--memory 1Gi` — headroom for OpenCV + pandas on the demo fixtures.
- Scale-to-zero is the default, so an idle demo costs ~nothing; the first
  request after idle cold-starts in a few seconds.

Demo-hardening notes:

- The API accepts server-side filesystem paths (by design, for local use).
  On a public URL those paths refer to the container's own ephemeral
  filesystem, so exposure is contained — but for anything beyond short
  demos, drop `--allow-unauthenticated` and open it with
  `gcloud run services proxy dashcam-demo --region us-central1` instead.
- The container filesystem is ephemeral: results vanish on scale-to-zero.
  Fine for demos; not a data store.

Tear down / pause:

```bash
gcloud run services delete dashcam-demo --region us-central1   # remove
# (or just leave it — scaled to zero it accrues no compute charges)
```

# Vertex AI batch jobs via GCP_Cloud_Send

This repo is deployable as a Vertex AI Custom Job through the
**GCP_Cloud_Send** desktop app, using the same pattern as the `cctv_zarr`
model: the app tars the repo, stages it to the GCS staging bucket, extracts
it into `/workspace` inside the training container, installs a requirements
file, and runs a single shell command. Nothing in GCP_Cloud_Send needs to
change — everything the flow expects lives here.

## Deployment surface (what maps to what)

| GCP_Cloud_Send field | Value for this repo |
|---|---|
| Code path | the repo root (`Cloud_Dashcam_POC/`) |
| Requirements file | `requirements.txt` (headless: `opencv-python-headless`, no UI extras) |
| Command | `bash scripts/vertex_entrypoint.sh` |
| Docker image (recommended) | a Python ≥ 3.11 training image with gsutil — build/push the included `Dockerfile`, or pick a `py311`-tagged Vertex prebuilt training image |
| Machine type | any CPU type (e.g. `n1-standard-4`); the pipeline is CPU-only |

Note on images: the app's default training image may ship an older Python.
The pipeline requires ≥ 3.11, and the entrypoint fails fast with a clear
message if the container is older. The image must include `gsutil`, since
GCP_Cloud_Send's generated setup command uses it to fetch the staged source.

## Job modes

`scripts/vertex_entrypoint.sh` picks a mode from job environment variables:

- **Smoke run (default, no variables set)** — generates synthetic dashcam
  footage + a GPX track and runs the full pipeline offline via
  `scripts/mock_dashcam_input.py` (the dashcam analog of cctv_zarr's
  `scripts/mock_lens_input.py`). Proves the deployment path end-to-end with
  no data dependencies.
- **Real run** — set `DASHCAM_VIDEOS_URI` to a `gs://` prefix containing
  footage (named `VEHICLE_YYYYMMDD_HHMMSS_CAMERA.ext`), and optionally
  `DASHCAM_GPX_URI` to a `gs://` GPX track. The entrypoint fetches both and
  runs `frame_extract.cli` against `config.yaml`.

In both modes the entrypoint copies results (kept frames, reflectance,
`frame_manifest.csv`, `video_summaries.csv`, `kept_slices.yaml`,
`coverage_chunks.yaml`) to the job's GCS output directory
(`AIP_MODEL_DIR`, under `gs://<staging>/vertex-jobs/<job>/`), so they are
retrievable from GCP_Cloud_Send's job monitor **download outputs** action,
and logs appear in its Cloud Logging view.

## Splash launcher (Local / Cloud toggle)

`frame-extract-splash` (default http://127.0.0.1:8320) serves a splash
screen with a Local/Cloud toggle:

- **Local** — starts the existing web control panel in the background and
  hands the browser to it. No credentials needed.
- **Cloud** — submits this repo as a Vertex AI Custom Job directly from the
  splash (project, region, staging bucket, container image, machine type;
  optional `gs://` footage + GPX URIs — leave empty for the offline mock
  smoke run). Uses Application Default Credentials and the same
  tar-and-stage pattern GCP_Cloud_Send uses; jobs submitted here appear in
  GCP_Cloud_Send's job monitor too. Requires the cloud extra:
  `pip install -e ".[cloud]"`.

Note: Google's prebuilt Vertex training images currently top out at
Python 3.10, below this repo's ≥ 3.11 floor — so the container image field
should point at the image built from the included `Dockerfile` (or any
py311+ image with gsutil).

## Local rehearsal (no cloud)

```bash
pip install -r requirements.txt
python scripts/mock_dashcam_input.py --scene all --output mock_dashcam_out
```

This is exactly what the Vertex smoke job runs, and what CI's `mock-input`
job runs on every push (`.github/workflows/ci.yml`). A linked pipeline in a
sibling repo (the way GCP_Cloud_Send's CI runs the cctv mock-lens stage) can
check out this repo and invoke the same two commands.

## Troubleshooting

**`cryptography` fails to build during `uv sync` / install** (maturin,
"Rust not found", "Could not find OpenSSL", pkg-config errors) — the `gcs`
extra pulls `google-cloud-storage` → `google-auth` → `cryptography`, which
has no prebuilt wheel for some platforms (notably Intel macOS on new
CPython versions), so the installer falls back to a Rust source build.

- `google-cloud-storage` is **not needed locally** — mock runs use the
  built-in mock GCS client, and Linux wheels exist for CI/Vertex. Just
  skip the extra: `uv sync --extra coverage --extra ui --extra dev`.
- `pyproject.toml` sets `tool.uv.no-build-package = ["cryptography"]`, so
  uv resolves to the newest cryptography that ships a wheel for your
  platform instead of compiling. (Equivalent flag:
  `uv sync --no-build-package cryptography`.)
- Last resort, to allow the source build:
  `brew install pkgconf openssl@3` then re-run with
  `OPENSSL_DIR="$(brew --prefix openssl@3)"`.
- Prefer Python 3.12/3.13 in `.python-version`; very new CPythons are the
  most likely to be missing wheels across the ecosystem.
