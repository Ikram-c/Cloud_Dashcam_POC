"""GCS download of the GPX track, with an injectable offline mock."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

_MOCK_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="MockGCP">
  <trk><trkseg>
    <trkpt lat="53.801" lon="-1.554"><time>2026-07-11T12:00:00Z</time></trkpt>
    <trkpt lat="53.805" lon="-1.545"><time>2026-07-11T12:00:05Z</time></trkpt>
    <trkpt lat="53.810" lon="-1.540"><time>2026-07-11T12:00:10Z</time></trkpt>
  </trkseg></trk>
</gpx>
"""


class MockBlob:
    """Stands in for a GCS blob; writes a minimal valid GPX file."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def download_to_filename(self, destination_file_name: str):
        """Write the synthetic GPX file.

        Args:
            destination_file_name (str): Local target path.
        """
        logger.info("[mock gcs] writing synthetic GPX for blob %s", self.name)
        Path(destination_file_name).write_text(_MOCK_GPX, encoding="utf-8")


class MockBucket:
    """Stands in for a GCS bucket."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def blob(self, blob_name: str) -> MockBlob:
        return MockBlob(blob_name)


class MockStorageClient:
    """Drop-in replacement for google.cloud.storage.Client."""

    def bucket(self, bucket_name: str) -> MockBucket:
        return MockBucket(bucket_name)


def get_storage_client(use_mock: bool = False):
    """Return a real or mock storage client.

    The real client picks up credentials from the standard
    GOOGLE_APPLICATION_CREDENTIALS environment variable.

    Args:
        use_mock (bool): Return the offline mock instead.

    Returns:
        A client exposing bucket(name).blob(name).download_to_filename.

    Raises:
        ImportError: If the real client is requested but the
            google-cloud-storage package is not installed.
    """
    if use_mock:
        return MockStorageClient()
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required unless use_mock=True")
    return storage.Client()


def download_gpx_track(
    client, bucket_name: str, blob_name: str, local_path: Path,
) -> Path:
    """Download the GPX track using the injected client.

    Args:
        client: Real or mock storage client.
        bucket_name (str): Bucket name.
        blob_name (str): Blob name.
        local_path (Path): Local destination.

    Returns:
        Path: The local path, for chaining.
    """
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    return Path(local_path)