FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libmediainfo0v5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY src ./src
COPY config.docker.yaml ./config.yaml

RUN pip install --no-cache-dir ".[metadata,ui,coverage]"

VOLUME ["/data"]
EXPOSE 8321

CMD ["frame-extract-ui", "--config", "config.yaml"]
