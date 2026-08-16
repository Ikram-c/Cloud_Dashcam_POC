# Optional custom training image for the GCP_Cloud_Send "Docker Image"
# environment option. Provides Python 3.11 + gsutil (required: the app's
# generated job command fetches the staged source tarball with gsutil)
# with the headless requirements preinstalled.
#
#   docker build -t REGION-docker.pkg.dev/PROJECT/REPO/dashcam-poc:latest .
#   docker push  REGION-docker.pkg.dev/PROJECT/REPO/dashcam-poc:latest
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
