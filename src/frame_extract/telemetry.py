"""Single source of truth for GPX telemetry resolution.

Reconstructed module: the original was lost from the repository.
One provider resolves the GPX track once per run (local path first,
then GCS - real or mock) and caches the parsed node list, so
illumination and coverage can never disagree about vehicle position.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .cloud import download_gpx_track, get_storage_client
from .config import Settings
from .gpx_telemetry import parse_gpx_to_nodes

logger = logging.getLogger(__name__)


class TelemetryProvider:
    """Resolves and caches the shared telemetry node list."""

    def __init__(self, settings: Settings, storage_client=None):
        """Store the telemetry config and optional injected client.

        Args:
            settings (Settings): Root configuration.
            storage_client: Injected GCS client (real or mock) for
                offline tests; created lazily from settings when None.
        """
        self.config = settings.telemetry
        self._client = storage_client
        self._nodes: Optional[List[dict]] = None

    @property
    def configured(self) -> bool:
        """Whether any telemetry source is configured.

        Returns:
            bool: True if a local path or a complete GCS source is set.
        """
        if self.config.gpx_path is not None:
            return True
        return (
            self.config.gcs_bucket is not None
            and self.config.gcs_blob is not None
        )

    def nodes(self, output_root: Path) -> List[dict]:
        """Resolve the telemetry nodes, downloading at most once.

        A local ``gpx_path`` wins over GCS. The GCS download lands
        under ``output_root`` so runs are self-contained.

        Args:
            output_root (Path): Directory for the downloaded track.

        Returns:
            List[dict]: Nodes with 'coords' (lon, lat) and aware 'time'.

        Raises:
            ValueError: If no telemetry source is configured.
            ImportError: If a required client or parser is missing.
            FileNotFoundError: If a local GPX path does not exist.
        """
        if self._nodes is not None:
            return self._nodes
        if self.config.gpx_path is not None:
            gpx_path = Path(self.config.gpx_path)
            logger.info("Telemetry: local GPX %s", gpx_path)
        elif self.config.gcs_bucket is not None and self.config.gcs_blob is not None:
            client = self._client
            if client is None:
                client = get_storage_client(use_mock=self.config.use_mock_gcs)
            gpx_path = Path(output_root) / Path(self.config.gcs_blob).name
            logger.info(
                "Telemetry: GCS gs://%s/%s -> %s",
                self.config.gcs_bucket, self.config.gcs_blob, gpx_path,
            )
            download_gpx_track(
                client, self.config.gcs_bucket, self.config.gcs_blob, gpx_path,
            )
        else:
            raise ValueError(
                "telemetry is not configured: set gpx_path or gcs_bucket/gcs_blob"
            )
        self._nodes = parse_gpx_to_nodes(gpx_path)
        return self._nodes
