"""Typed YAML configuration loading into frozen dataclasses.

Every numeric parameter originates from ``config.yaml``; module code
contains no magic numbers (NASA Power of 10, rule 8).
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import CropConfig, SampleConfig


def _to_tuple(value: Any) -> Any:
    """Convert YAML lists to tuples recursively for immutability.

    Args:
        value (Any): A value parsed from YAML.

    Returns:
        Any: The value with all lists converted to tuples.
    """
    if isinstance(value, list):
        return tuple(_to_tuple(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_tuple(v) for k, v in value.items()}
    return value


def _build(cls, section: Optional[dict]):
    """Instantiate a frozen dataclass from a YAML section.

    Args:
        cls: The dataclass type to instantiate.
        section (Optional[dict]): Raw YAML mapping for this section.

    Returns:
        An instance of ``cls``.

    Raises:
        KeyError: If the section is missing or has unknown keys.
    """
    if section is None:
        raise KeyError(f"Missing config section for {cls.__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise KeyError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _to_tuple(v) for k, v in section.items()})


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Frame sampling strategy and bounds."""

    mode: str
    every_n_frames: int
    every_n_seconds: float
    max_frames_per_video: int

    def __post_init__(self):
        if self.mode not in ("frames", "seconds"):
            raise ValueError("mode must be 'frames' or 'seconds'")
        if self.every_n_frames <= 0:
            raise ValueError("every_n_frames must be positive")
        if self.every_n_seconds <= 0.0:
            raise ValueError("every_n_seconds must be positive")
        if self.max_frames_per_video <= 0:
            raise ValueError("max_frames_per_video must be positive")


@dataclass(frozen=True, slots=True)
class InputConfig:
    """Pre-gate input handling: crop, sample window, metadata source.

    Raw YAML mappings for crop/sample are promoted to typed frozen
    models at load time, so their values validate fail-fast and no
    mutable dict survives inside the frozen tree.
    """

    crop: Optional[CropConfig]
    sample: Optional[SampleConfig]
    metadata_source: str

    def __post_init__(self):
        if isinstance(self.crop, dict):
            object.__setattr__(self, "crop", CropConfig(**self.crop))
        if isinstance(self.sample, dict):
            object.__setattr__(self, "sample", SampleConfig(**self.sample))
        if self.metadata_source not in ("auto", "container", "filename", "none"):
            raise ValueError("metadata_source must be auto|container|filename|none")


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Quality gate thresholds, all terrain-agnostic."""

    enabled: bool
    blur_threshold: float
    min_mean_intensity: float
    max_mean_intensity: float
    duplicate_threshold: float
    analysis_size: tuple

    def __post_init__(self):
        if self.blur_threshold < 0.0:
            raise ValueError("blur_threshold must be non-negative")
        if self.min_mean_intensity >= self.max_mean_intensity:
            raise ValueError("min_mean_intensity must be below max_mean_intensity")
        if self.duplicate_threshold < 0.0:
            raise ValueError("duplicate_threshold must be non-negative")
        if len(self.analysis_size) != 2:
            raise ValueError("analysis_size must be (width, height)")
        if self.analysis_size[0] <= 0 or self.analysis_size[1] <= 0:
            raise ValueError("analysis_size must be positive")


@dataclass(frozen=True, slots=True)
class IntrinsicConfig:
    """Weiss (2001) intrinsic-image recovery over static segments."""

    enabled: bool
    min_frames: int
    max_frames: int
    log_epsilon: float

    def __post_init__(self):
        if self.min_frames < 3:
            raise ValueError("min_frames must be >= 3 (median needs a majority)")
        if self.max_frames < self.min_frames:
            raise ValueError("max_frames must be >= min_frames")
        if self.log_epsilon <= 0.0:
            raise ValueError("log_epsilon must be positive")


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """Shared GPX telemetry source for illumination and coverage.

    Source completeness is validated at first use by the provider,
    not here, because whether telemetry is *required* depends on
    sibling sections this frozen dataclass cannot see.
    """

    gpx_path: Optional[str]
    gcs_bucket: Optional[str]
    gcs_blob: Optional[str]
    use_mock_gcs: bool


@dataclass(frozen=True, slots=True)
class IlluminationConfig:
    """Forecast-prior and probe-based illumination estimation."""

    enabled: bool
    timezone: str
    fixed_lat: Optional[float]
    fixed_lon: Optional[float]
    gnss_tolerance_s: float
    forecast_provider: str
    forecast_url: str
    forecast_archive_url: str
    forecast_max_past_days: int
    http_timeout_s: int
    cache_path: str
    clear_sky_log_intensity: float
    night_log_intensity: float
    cloud_attenuation_log: float
    twilight_elevation_deg: float
    min_sun_factor: float
    probe: str
    ground_patch_frac: tuple
    vehicle_model_prototxt: Optional[str]
    vehicle_model_weights: Optional[str]
    vehicle_confidence: float
    vehicle_iou_min: float
    vehicle_min_track_frames: int
    fusion_gain: float
    exposure_tolerance_stops: float
    confidence_base: float
    confidence_per_probe: float
    normalize_output: bool

    def __post_init__(self):
        if self.forecast_provider not in ("open_meteo", "none"):
            raise ValueError("forecast_provider must be 'open_meteo' or 'none'")
        if self.forecast_max_past_days <= 0:
            raise ValueError("forecast_max_past_days must be positive")
        if self.probe not in ("ground", "vehicle", "both", "none"):
            raise ValueError("probe must be ground|vehicle|both|none")
        if self.night_log_intensity >= self.clear_sky_log_intensity:
            raise ValueError("night_log_intensity must be below clear_sky level")
        if not 0.0 < self.fusion_gain <= 1.0:
            raise ValueError("fusion_gain must be in (0, 1]")
        if self.exposure_tolerance_stops <= 0.0:
            raise ValueError("exposure_tolerance_stops must be positive")
        if not 0.0 <= self.confidence_base <= 1.0:
            raise ValueError("confidence_base must be in [0, 1]")
        if self.confidence_per_probe < 0.0:
            raise ValueError("confidence_per_probe must be non-negative")
        if len(self.ground_patch_frac) != 4:
            raise ValueError("ground_patch_frac must be (x1, y1, x2, y2)")
        x1, y1, x2, y2 = self.ground_patch_frac
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("ground_patch_frac must be ordered fractions in [0, 1]")


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """Network-coverage chunking: carriers and their zone maps."""

    enabled: bool
    network: str
    networks: dict
    chunk_manifest_name: str

    def __post_init__(self):
        if not self.enabled:
            return
        if not self.networks:
            raise ValueError("coverage requires at least one network")
        if self.network not in self.networks:
            raise ValueError(
                f"network '{self.network}' not in networks: {sorted(self.networks)}"
            )
        for net, zones in self.networks.items():
            if not zones:
                raise ValueError(f"network {net}: requires at least one zone")
            for name, bbox in zones.items():
                if len(bbox) != 4:
                    raise ValueError(f"{net}/{name}: bbox must be (x0, y0, x1, y1)")
                x0, y0, x1, y1 = bbox
                if x0 >= x1 or y0 >= y1:
                    raise ValueError(f"{net}/{name}: bbox must be ordered min < max")

    @property
    def zones(self) -> dict:
        """The selected network's zone map."""
        return self.networks[self.network]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Image export format, manifest, and bridge settings."""

    image_format: str
    jpeg_quality: int
    resize_width: Optional[int]
    manifest_name: str
    export_kept_slices: bool

    def __post_init__(self):
        if self.image_format not in ("png", "jpg"):
            raise ValueError("image_format must be 'png' or 'jpg'")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.resize_width is not None and self.resize_width <= 0:
            raise ValueError("resize_width must be positive when set")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Paths, bounds, parallelism, and behaviour flags."""

    video_directory: str
    output_directory: str
    video_extensions: tuple
    default_fps: float
    max_video_frames: int
    max_video_files: int
    max_consecutive_fails: int
    num_workers: int
    parse_filenames: bool

    def __post_init__(self):
        if self.default_fps <= 0.0:
            raise ValueError("default_fps must be positive")
        if self.max_video_frames <= 0:
            raise ValueError("max_video_frames must be positive")
        if self.max_video_files <= 0:
            raise ValueError("max_video_files must be positive")
        if self.max_consecutive_fails <= 0:
            raise ValueError("max_consecutive_fails must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True, slots=True)
class UIConfig:
    """Web UI host, port, and thumbnail settings."""

    host: str
    port: int
    thumbnail_width: int

    def __post_init__(self):
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.thumbnail_width <= 0:
            raise ValueError("thumbnail_width must be positive")


@dataclass(frozen=True, slots=True)
class Settings:
    """Root configuration aggregating all typed sections."""

    sampling: SamplingConfig
    input: InputConfig
    quality: QualityConfig
    intrinsic: IntrinsicConfig
    telemetry: TelemetryConfig
    illumination: IlluminationConfig
    coverage: CoverageConfig
    output: OutputConfig
    runtime: RuntimeConfig
    ui: UIConfig

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load and validate settings from a YAML file.

        Args:
            path (Path): Path to ``config.yaml``.

        Returns:
            Settings: A fully validated, immutable settings object.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If a section or key is missing or unknown.
            ValueError: If any value fails validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")
        return cls(
            sampling=_build(SamplingConfig, raw.get("sampling")),
            input=_build(InputConfig, raw.get("input")),
            quality=_build(QualityConfig, raw.get("quality")),
            intrinsic=_build(IntrinsicConfig, raw.get("intrinsic")),
            telemetry=_build(TelemetryConfig, raw.get("telemetry")),
            illumination=_build(IlluminationConfig, raw.get("illumination")),
            coverage=_build(CoverageConfig, raw.get("coverage")),
            output=_build(OutputConfig, raw.get("output")),
            runtime=_build(RuntimeConfig, raw.get("runtime")),
            ui=_build(UIConfig, raw.get("ui")),
        )

    def with_overrides(
        self,
        video_directory: Optional[Path] = None,
        output_directory: Optional[Path] = None,
        gpx: Optional[Path] = None,
    ) -> "Settings":
        """Return a copy with CLI overrides applied.

        Args:
            video_directory (Optional[Path]): Video directory override.
            output_directory (Optional[Path]): Output directory override.
            gpx (Optional[Path]): GPX telemetry file override.

        Returns:
            Settings: A new settings object with overrides applied.
        """
        runtime = self.runtime
        telemetry = self.telemetry
        if video_directory is not None:
            runtime = replace(runtime, video_directory=str(video_directory))
        if output_directory is not None:
            runtime = replace(runtime, output_directory=str(output_directory))
        if gpx is not None:
            telemetry = replace(telemetry, gpx_path=str(gpx))
        return replace(self, runtime=runtime, telemetry=telemetry)