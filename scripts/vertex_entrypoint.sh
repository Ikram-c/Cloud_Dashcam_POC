#!/usr/bin/env bash
# Vertex AI Custom Job entrypoint for Cloud_Dashcam_POC.
#
# Designed for the GCP_Cloud_Send deployment flow (same as the cctv_zarr
# model): the app tars this repo, extracts it into /workspace inside the
# training container, optionally pip-installs requirements.txt, then runs
# the job command. Use this script as that command:
#
#   bash scripts/vertex_entrypoint.sh
#
# Modes (selected via job environment variables):
#   DASHCAM_VIDEOS_URI  gs:// prefix of real footage  -> real extraction run
#   DASHCAM_GPX_URI     gs:// URI of a GPX track      -> telemetry for the run
#   (neither set)                                     -> offline mock smoke run
#
# Outputs are copied to the job's GCS output directory (AIP_MODEL_DIR is
# set by Vertex under gs://<staging>/vertex-jobs/<job>/), so they are
# retrievable from GCP_Cloud_Send's job monitor "download outputs" action.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${DASHCAM_OUTPUT_DIR:-vertex_run_out}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

# The pipeline targets Python >= 3.11; fail fast with a clear message if the
# chosen training image is older (pick a py311+ image in GCP_Cloud_Send's
# "Docker Image" field — see DEPLOY.md).
python - <<'EOF'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Cloud_Dashcam_POC needs Python >= 3.11; container has "
             f"{sys.version.split()[0]}. Choose a newer training image.")
EOF

if [ -n "${DASHCAM_VIDEOS_URI:-}" ]; then
    echo "[entrypoint] real run: fetching footage from ${DASHCAM_VIDEOS_URI}"
    mkdir -p input/videos
    gsutil -m cp -r "${DASHCAM_VIDEOS_URI%/}/*" input/videos/
    GPX_ARGS=()
    if [ -n "${DASHCAM_GPX_URI:-}" ]; then
        gsutil cp "${DASHCAM_GPX_URI}" input/track.gpx
        GPX_ARGS=(--gpx input/track.gpx)
    fi
    python -m frame_extract.cli --config config.yaml \
        --videos input/videos --output "${OUT_DIR}" \
        ${GPX_ARGS[@]+"${GPX_ARGS[@]}"}
else
    echo "[entrypoint] no DASHCAM_VIDEOS_URI set: running offline mock smoke"
    python scripts/mock_dashcam_input.py --scene all --output "${OUT_DIR}"
fi

if [ -n "${AIP_MODEL_DIR:-}" ] && command -v gsutil >/dev/null 2>&1; then
    echo "[entrypoint] staging outputs to ${AIP_MODEL_DIR}"
    gsutil -m cp -r "${OUT_DIR}" "${AIP_MODEL_DIR%/}/"
else
    echo "[entrypoint] AIP_MODEL_DIR not set; outputs left in ${OUT_DIR}"
fi
echo "[entrypoint] done"
