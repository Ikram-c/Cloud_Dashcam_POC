"""Core offline tests: config, scanner, gates, extractor, pipeline."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from frame_extract import (
    ExtractionPipeline, FrameExtractor, FrameQualityGate, Settings,
    VideoScanner,
)
from frame_extract.models import ExtractedFrame, SampleConfig
from frame_extract.pipeline import ExtractionPipeline as Pipeline

CONFIG = Path(__file__).parent.parent / "config.yaml"
TEST_DATA = Path(__file__).parent.parent / "test_data"


@pytest.fixture(scope="session", autouse=True)
def fixtures():
    """Generate test data once per session if absent."""
    if not (TEST_DATA / "videos").exists():
        subprocess.run(
            [sys.executable, "scripts/make_test_data.py"],
            check=True, cwd=Path(__file__).parent.parent,
        )
    yield


@pytest.fixture
def settings(tmp_path):
    """Offline settings: fixture videos, temp output, no network."""
    base = Settings.load(CONFIG).with_overrides(
        video_directory=TEST_DATA / "videos",
        output_directory=tmp_path,
    )
    illumination = replace(
        base.illumination,
        forecast_provider="none",
        cache_path=str(tmp_path / "cache.json"),
    )
    return replace(base, illumination=illumination)


def _sharp_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _flat_frame(value: int) -> np.ndarray:
    return np.full((240, 320, 3), value, dtype=np.uint8)


class TestConfig:
    def test_load_valid(self, settings):
        assert settings.quality.analysis_size == (320, 240)
        assert settings.intrinsic.min_frames == 5

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Settings.load(Path("no_such_config.yaml"))

    def test_frozen(self, settings):
        with pytest.raises(AttributeError):
            settings.quality.blur_threshold = 0.0

    def test_unknown_key_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        text = CONFIG.read_text().replace("mode: seconds", "mode: seconds\n  bogus: 1")
        bad.write_text(text)
        with pytest.raises(KeyError):
            Settings.load(bad)

    def test_input_sections_typed(self, settings):
        """Crop/sample must be frozen models, not mutable dicts."""
        assert not isinstance(settings.input.crop, dict)
        assert not isinstance(settings.input.sample, dict)

    def test_gpx_override(self, settings):
        out = settings.with_overrides(gpx=Path("track.gpx"))
        assert out.telemetry.gpx_path == "track.gpx"


class TestScanner:
    def test_finds_videos(self, settings):
        assert len(VideoScanner(settings.runtime).scan()) == 5

    def test_bad_filename_returns_none(self, settings):
        scanner = VideoScanner(settings.runtime)
        assert scanner.parse_filename(Path("not_a_valid_name.mp4")) is None

    def test_good_filename_parses(self, settings):
        scanner = VideoScanner(settings.runtime)
        parsed = scanner.parse_filename(Path("CAR01_20240315_083000_CAM01.mp4"))
        assert parsed["vehicle"] == "CAR01"
        assert parsed["camera_id"] == "CAM01"


class TestQualityGate:
    def test_sharp_frame_accepted(self, settings):
        gate = FrameQualityGate(settings.quality)
        assert gate.evaluate(_sharp_frame()).accepted

    def test_blurred_frame_rejected(self, settings):
        import cv2
        gate = FrameQualityGate(settings.quality)
        blurred = cv2.GaussianBlur(_flat_frame(120), (51, 51), 15)
        result = gate.evaluate(blurred)
        assert not result.accepted
        assert result.reject_reason == "blur"

    def test_dark_frame_rejected_static(self, settings):
        gate = FrameQualityGate(settings.quality)
        result = gate.evaluate(_sharp_frame() // 30)
        assert result.reject_reason in ("underexposed", "blur")

    def test_duplicate_survives_global_gain(self, settings):
        """Exposure change alone must not defeat duplicate detection."""
        gate = FrameQualityGate(settings.quality)
        base = _sharp_frame()
        gate.evaluate(base)
        gained = np.clip(base.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        result = gate.evaluate(gained)
        assert result.reject_reason == "duplicate"

    def test_band_uses_mean_of_log(self, settings):
        """The band check must use mean(log), not log(mean)."""
        gate = FrameQualityGate(settings.quality)
        frame = _sharp_frame()
        gray = np.mean(frame, axis=2)
        mean_of_log = float(np.mean(np.log(gray.astype(np.float32) + 1.0)))
        log_of_mean = float(np.log(np.mean(gray) + 1.0))
        assert log_of_mean - mean_of_log > 0.2
        band = (mean_of_log - 0.15, mean_of_log + 0.15)
        result = gate.evaluate(frame, exposure_band=band)
        assert result.reject_reason not in ("underexposed", "overexposed")

    def test_baseline_updates_on_rejected_frames(self, settings):
        """A blur rejection between duplicates must not stale the baseline."""
        import cv2
        gate = FrameQualityGate(settings.quality)
        base = _sharp_frame()
        gate.evaluate(base)
        blurred = cv2.GaussianBlur(_flat_frame(120), (51, 51), 15)
        gate.evaluate(blurred)
        result = gate.evaluate(base)
        assert result.reject_reason != "duplicate"


class TestExtractor:
    def test_pts_fallback(self, settings, tmp_path):
        """Zero PTS past frame 0 must trigger the index/fps fallback,
        retroactively unifying earlier frames onto the same time base."""
        extractor = FrameExtractor(settings)
        video = TEST_DATA / "videos" / "CAR01_20240315_083000_CAM01.mp4"
        with patch("frame_extract.extractor.TimestampedFrameIterator") as it:
            frames = [(0, 0.0, _sharp_frame()), (30, 0.0, _sharp_frame())]
            it.return_value = iter(frames)
            kept, stats = extractor.extract(video, tmp_path / "out")
        assert len(kept) == 2
        fps = stats["fps"]
        assert kept[0].timestamp_sec == pytest.approx(0.0)
        assert kept[-1].timestamp_sec == pytest.approx(30 / fps)
        assert all(f.timestamp_source == "index_fps" for f in kept)

    def test_frame_cap_bounds_count_not_end_index(self, settings, tmp_path):
        """A sample window starting past the cap must not raise, and must
        get its own frame budget (regression for FrameSlice(start > end))."""
        runtime = replace(settings.runtime, max_video_frames=10)
        inp = replace(settings.input, sample=SampleConfig(
            duration_seconds=2.0, offset_seconds=3.0,   # start=90 @ 30 fps
        ))
        cfg = replace(settings, runtime=runtime, input=inp)
        extractor = FrameExtractor(cfg)
        video = TEST_DATA / "videos" / "CAR01_20240315_083000_CAM01.mp4"
        kept, stats = extractor.extract(video, tmp_path / "cap_out")
        assert stats["slice_frames"] == 10
        assert all(90 <= f.frame_index < 100 for f in kept)

    def test_blurred_video_rejects_most(self, settings, tmp_path):
        extractor = FrameExtractor(settings)
        video = TEST_DATA / "videos" / "CAR01_20240315_103000_CAM01.mp4"
        kept, stats = extractor.extract(video, tmp_path / "blur_out")
        assert stats["blur"] > len(kept)

    def test_static_video_yields_duplicates(self, settings, tmp_path):
        extractor = FrameExtractor(settings)
        video = TEST_DATA / "videos" / "CAR01_20240315_120000_CAM01.mp4"
        kept, stats = extractor.extract(video, tmp_path / "static_out")
        assert stats["duplicate"] >= 1

    def test_total_frames_is_whole_video(self, settings, tmp_path):
        extractor = FrameExtractor(settings)
        video = TEST_DATA / "videos" / "CAR01_20240315_083000_CAM01.mp4"
        _, stats = extractor.extract(video, tmp_path / "tf_out")
        assert stats["total_frames"] >= stats["slice_frames"]


class TestPipeline:
    def test_full_run_offline(self, settings, tmp_path):
        summaries = ExtractionPipeline(settings).run()
        assert len(summaries) >= 4
        assert (tmp_path / settings.output.manifest_name).exists()
        assert (tmp_path / "video_summaries.csv").exists()

    def test_progress_events_emitted(self, settings):
        events = []
        ExtractionPipeline(settings).run(progress=events.append)
        assert events
        assert all(e.video_count == len({e.video_path for e in events})
                   or e.video_count >= 1 for e in events)

    def test_kept_slices_collapsing(self):
        frames = [
            ExtractedFrame(Path("v.mp4"), i, i / 30.0, Path("f.png"), 100.0, 120.0)
            for i in (0, 30, 60, 150, 180)
        ]
        ranges = Pipeline._kept_slices(frames, every=30)
        assert ranges == [{"start": 0, "end": 61}, {"start": 150, "end": 181}]

    def test_effective_every_inferred(self):
        frames = [
            ExtractedFrame(Path("v.mp4"), i, 0.0, Path("f.png"), 0.0, 0.0)
            for i in (0, 30, 90)
        ]
        assert Pipeline._effective_every(frames) == 30