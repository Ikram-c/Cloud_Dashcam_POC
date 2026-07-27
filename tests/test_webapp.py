"""Web control panel API tests. Pipeline mocked; fully offline."""

import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from frame_extract.config import Settings
from frame_extract.webapp.server import (
    ExtractRequest, _apply_request, create_app,
)

CONFIG = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture
def client():
    return TestClient(create_app(CONFIG))


class TestStatusEndpoint:
    def test_initial_state_is_idle(self, client):
        body = client.get("/api/status").json()
        assert body["state"] == "idle"
        assert body["progress"] == 0.0

    def test_snapshot_carries_network_and_zones(self, client):
        body = client.get("/api/status").json()
        assert "network" in body and "zones" in body

    def test_index_serves_flow_screens(self, client):
        text = client.get("/").text
        for marker in ("Cloud Optimised DashCam", "Run Local Demo",
                       "Select Video File", "Processing Complete"):
            assert marker in text


class TestNetworksEndpoint:
    def test_networks_lists_default(self, client):
        body = client.get("/api/networks").json()
        assert body["default"] in body["networks"]


class TestExtractEndpoint:
    def test_launch_and_complete(self, client):
        with patch(
            "frame_extract.webapp.server.ExtractionPipeline"
        ) as pipeline_cls, patch(
            "frame_extract.webapp.server._read_zone_counts", return_value={},
        ):
            pipeline_cls.return_value.run.return_value = []
            res = client.post("/api/extract", json={"gates": {}})
            assert res.status_code == 200
            state = "running"
            deadline = time.time() + 5.0
            while time.time() < deadline:
                state = client.get("/api/status").json()["state"]
                if state in ("done", "failed"):
                    break
                time.sleep(0.05)
            assert state == "done"

    def test_concurrent_job_rejected(self, client):
        with patch(
            "frame_extract.webapp.server.ExtractionPipeline"
        ) as pipeline_cls, patch(
            "frame_extract.webapp.server._read_zone_counts", return_value={},
        ):
            pipeline_cls.return_value.run.side_effect = (
                lambda progress=None: time.sleep(0.5) or []
            )
            first = client.post("/api/extract", json={"gates": {}})
            second = client.post("/api/extract", json={"gates": {}})
        assert first.status_code == 200
        assert second.status_code == 409

    def test_bad_config_reports_failed(self, tmp_path):
        client = TestClient(create_app(tmp_path / "missing.yaml"))
        res = client.post("/api/extract", json={"gates": {}})
        assert res.status_code == 400
        assert client.get("/api/status").json()["state"] == "failed"

    def test_unknown_network_is_400(self, client):
        res = client.post(
            "/api/extract", json={"network": "carrier_pigeon", "gates": {}},
        )
        assert res.status_code == 400


class TestApplyRequest:
    def test_toggles_neutralise_thresholds(self):
        settings = Settings.load(CONFIG)
        req = ExtractRequest(gates={
            "blur": False, "exposure": False, "duplicate": False,
            "illumination": False, "weiss": False, "coverage": False,
        })
        out = _apply_request(settings, req)
        assert out.quality.blur_threshold == 0.0
        assert out.quality.max_mean_intensity == 256.0
        assert out.quality.duplicate_threshold == 0.0
        assert not out.illumination.enabled
        assert not out.intrinsic.enabled
        assert not out.coverage.enabled

    def test_toggle_cannot_enable_disabled_subsystem(self):
        settings = Settings.load(CONFIG)
        settings = replace(
            settings, intrinsic=replace(settings.intrinsic, enabled=False),
        )
        out = _apply_request(settings, ExtractRequest(gates={"weiss": True}))
        assert not out.intrinsic.enabled

    def test_gpx_source_maps_to_telemetry(self):
        settings = Settings.load(CONFIG)
        settings = replace(
            settings, telemetry=replace(
                settings.telemetry,
                gpx_path="local.gpx", gcs_bucket="b", gcs_blob="t.gpx",
            ),
        )
        out = _apply_request(
            settings, ExtractRequest(gpx_source="mock", gates={}),
        )
        assert out.telemetry.use_mock_gcs
        assert out.telemetry.gpx_path is None

    def test_network_selection_applied(self):
        settings = Settings.load(CONFIG)
        out = _apply_request(
            settings, ExtractRequest(network="o2", gates={}),
        )
        assert out.coverage.network == "o2"

    def test_sampling_choice_applied(self):
        settings = Settings.load(CONFIG)
        out = _apply_request(
            settings, ExtractRequest(sampling="frames_30", gates={}),
        )
        assert out.sampling.mode == "frames"
        assert out.sampling.every_n_frames == 30


def _tmp_config(tmp_path):
    """Write a config rooted at tmp_path for filesystem endpoints.

    Args:
        tmp_path: Test-scoped directory.

    Returns:
        Path: The written config path.
    """
    import yaml
    raw = yaml.safe_load(CONFIG.read_text())
    raw["runtime"]["video_directory"] = str(tmp_path / "videos")
    raw["runtime"]["output_directory"] = str(tmp_path / "out")
    (tmp_path / "videos").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def fs_client(tmp_path):
    return TestClient(create_app(_tmp_config(tmp_path))), tmp_path


class TestBrowseEndpoint:
    def test_lists_videos_and_folders(self, fs_client):
        client, tmp = fs_client
        (tmp / "videos" / "night").mkdir()
        (tmp / "videos" / "drive.mp4").write_bytes(b"x" * 64)
        body = client.post("/api/browse", json={}).json()
        assert "night" in body["folders"]
        assert [v["name"] for v in body["videos"]] == ["drive.mp4"]
        assert body["videos"][0]["processed"] is False

    def test_marks_processed_videos(self, fs_client):
        client, tmp = fs_client
        (tmp / "videos" / "done.mp4").write_bytes(b"x" * 64)
        (tmp / "out" / "done").mkdir(parents=True)
        body = client.post("/api/browse", json={}).json()
        assert body["videos"][0]["processed"] is True

    def test_missing_folder_is_400(self, fs_client):
        client, _ = fs_client
        res = client.post("/api/browse", json={"path": "/nowhere"})
        assert res.status_code == 400


class TestUploadEndpoint:
    def test_chunked_upload_roundtrip(self, fs_client):
        client, tmp = fs_client
        payload = b"fake video bytes" * 50
        import base64 as b64
        half = len(payload) // 2
        res = None
        for seq, part in enumerate((payload[:half], payload[half:])):
            res = client.post("/api/upload", json={
                "upload_id": "u1", "name": "picked.mp4", "seq": seq,
                "last": seq == 1,
                "data": b64.b64encode(part).decode("ascii"),
            })
        body = res.json()
        assert body["done"] is True
        assert Path(body["path"]).read_bytes() == payload

    def test_traversal_name_is_400(self, fs_client):
        client, _ = fs_client
        res = client.post("/api/upload", json={
            "upload_id": "u2", "name": "../evil.mp4", "seq": 0,
            "last": True, "data": "",
        })
        assert res.status_code == 400

    def test_non_video_is_400(self, fs_client):
        client, _ = fs_client
        res = client.post("/api/upload", json={
            "upload_id": "u3", "name": "notes.txt", "seq": 0,
            "last": True, "data": "",
        })
        assert res.status_code == 400


class TestSelectedRun:
    def test_selected_files_are_staged(self, fs_client):
        client, tmp = fs_client
        video = tmp / "videos" / "pick.mp4"
        video.write_bytes(b"x" * 64)
        with patch(
            "frame_extract.webapp.server.ExtractionPipeline"
        ) as pipeline_cls, patch(
            "frame_extract.webapp.server._read_zone_counts",
            return_value={},
        ):
            pipeline_cls.return_value.run.return_value = []
            res = client.post("/api/extract", json={
                "gates": {}, "selected": [str(video)],
            })
            assert res.status_code == 200
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if client.get("/api/status").json()["state"] in (
                    "done", "failed",
                ):
                    break
                time.sleep(0.05)
        staged = list((tmp / "out" / "_selected_run").iterdir())
        assert len(staged) == 1
        assert staged[0].name.endswith("pick.mp4")

    def test_missing_selected_is_400(self, fs_client):
        client, _ = fs_client
        res = client.post("/api/extract", json={
            "gates": {}, "selected": ["/nowhere/ghost.mp4"],
        })
        assert res.status_code == 400


class TestPrefs:
    def test_roundtrip(self, fs_client):
        client, _ = fs_client
        saved = client.post("/api/prefs", json={
            "prefs": {"tab": "options", "network": "o2"},
        }).json()
        assert saved["saved"] is True
        body = client.get("/api/prefs").json()
        assert body["prefs"]["tab"] == "options"
        assert body["prefs"]["network"] == "o2"

    def test_empty_when_unsaved(self, fs_client):
        client, _ = fs_client
        assert client.get("/api/prefs").json() == {"prefs": {}}


class TestPlainLanguage:
    def test_panel_markers(self, client):
        text = client.get("/").text
        for marker in ("Add footage", "Options", "Results",
                       "Choose files", "Drop videos here",
                       "Skip blurry frames", "tab-bar"):
            assert marker in text

    def test_guideline_rules_present(self, client):
        text = client.get("/").text
        for rule in ("-apple-system", "font-size: 17px",
                     "min-height: 44px",
                     "env(safe-area-inset-bottom)",
                     "border-radius: 22.5%",
                     "minmax(260px, 320px) 1fr",
                     "overscroll-behavior: contain",
                     "prefers-color-scheme: dark"):
            assert rule in text
