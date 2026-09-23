"""
Integration Tests for FastAPI Live Streaming & Visualization API (Phase 14 Part B).

Tests REST endpoints, WebSocket telemetry, session management, and spatial query routing.
"""

import pytest
import time
from fastapi.testclient import TestClient
import numpy as np

from backend.api.main import app
from backend.api.session import session_manager
from backend.data.frame import LidarFrame


@pytest.fixture(autouse=True)
def reset_session():
    """Ensure session manager is in clean stopped state before and after each test."""
    session_manager.reset()
    yield
    session_manager.reset()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestRestApiEndpoints:

    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["pipeline_ready"] is True
        assert "is_streaming" in data

    def test_root_dashboard_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        # Check that HTML content is served
        assert "Adaptive Variable-Resolution 2.5D LiDAR Mapping" in res.text

    def test_stream_status_initial(self, client):
        res = client.get("/api/stream/status")
        assert res.status_code == 200
        data = res.json()
        assert data["is_streaming"] is False
        assert data["frames_processed"] == 0
        assert data["has_active_map"] is False

    def test_metrics_endpoint(self, client):
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "pipeline_metrics" in data
        assert "current_fps" in data
        assert "has_active_map" in data

    def test_map_query_before_map_exists(self, client):
        res = client.post("/api/map/query", json={"x": 0.0, "y": 0.0})
        assert res.status_code == 200
        data = res.json()
        assert data["is_represented"] is False
        assert "No active AdaptiveMap25D" in data.get("message", "")

    def test_map_query_after_processing(self, client):
        # Manually process one frame through pipeline to populate map
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
            [0.1, 0.1, 1.6, 110.0],
            [1.0, 1.0, 0.5, 90.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        # Query center cell
        res = client.post("/api/map/query", json={"x": 0.0, "y": 0.0})
        assert res.status_code == 200
        data = res.json()
        assert data["is_represented"] is True
        assert data["tier"] is not None
        assert data["point_count"] > 0
        assert data["mean_z"] is not None

    def test_map_query_out_of_bounds(self, client):
        pts = np.array([[0.0, 0.0, 1.5, 100.0]], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        # Query far out-of-bounds coordinate
        res = client.post("/api/map/query", json={"x": 1000.0, "y": 1000.0})
        assert res.status_code == 200
        data = res.json()
        assert data["is_represented"] is False


class TestStreamingSessionManagement:

    def test_start_and_stop_streaming(self, client):
        # Start stream
        res_start = client.post("/api/stream/start", json={"fps": 30.0, "total_frames": 5})
        assert res_start.status_code == 200
        data_start = res_start.json()
        assert data_start["streaming"] is True
        assert data_start["status"] == "started"

        # Check status shows streaming
        res_status = client.get("/api/stream/status")
        assert res_status.json()["is_streaming"] is True

        # Duplicate start protection
        res_dup = client.post("/api/stream/start", json={"fps": 30.0})
        assert res_dup.status_code == 200
        assert res_dup.json()["status"] == "already_running"

        # Stop stream
        res_stop = client.post("/api/stream/stop")
        assert res_stop.status_code == 200
        data_stop = res_stop.json()
        assert data_stop["streaming"] is False
        assert data_stop["status"] == "stopped"

        # Check status shows stopped
        res_status2 = client.get("/api/stream/status")
        assert res_status2.json()["is_streaming"] is False


class TestWebSocketStreaming:

    def test_websocket_connection_and_ping(self, client):
        with client.websocket_connect("/ws/stream") as websocket:
            # First message on connect is connection_established
            init_msg = websocket.receive_json()
            assert init_msg["type"] == "connection_established"
            assert "is_streaming" in init_msg

            # Test ping command
            websocket.send_json({"command": "ping", "timestamp": 12345})
            resp = websocket.receive_json()
            assert resp["type"] == "pong"
            assert resp["timestamp"] == 12345

            # Test get_status command
            websocket.send_json({"command": "get_status"})
            resp_status = websocket.receive_json()
            assert resp_status["type"] == "status"
            assert "is_streaming" in resp_status["status"]


class TestDatasetReplayPlaybackEndpoints:

    def test_playback_step_forward_and_backward(self, client):
        """Verify stepping forward and backward advances frame playhead and returns valid status."""
        res_step1 = client.post("/api/stream/step", json={"direction": 1})
        assert res_step1.status_code == 200
        st1 = res_step1.json()
        assert st1["current_frame"] == 1
        assert "candidate_obstacles_count" in st1
        assert "terrain_risk" in st1

        res_step2 = client.post("/api/stream/step", json={"direction": -1})
        assert res_step2.status_code == 200
        st2 = res_step2.json()
        assert st2["current_frame"] == 0

    def test_playback_seek(self, client):
        """Verify seeking to frame index updates current_frame position."""
        res_seek = client.post("/api/stream/seek", json={"frame_idx": 3})
        assert res_seek.status_code == 200
        st = res_seek.json()
        assert st["current_frame"] == 3

    def test_playback_speed(self, client):
        """Verify speed multiplier adjustment."""
        res_speed = client.post("/api/stream/speed", json={"speed": 2.0})
        assert res_speed.status_code == 200
        st = res_speed.json()
        assert st["playback_speed"] == 2.0

    def test_playback_pause_and_reset(self, client):
        """Verify pause and reset operations."""
        client.post("/api/stream/seek", json={"frame_idx": 2})
        res_pause = client.post("/api/stream/pause")
        assert res_pause.status_code == 200
        assert res_pause.json()["playback_state"] == "PAUSED"

        res_reset = client.post("/api/stream/reset")
        assert res_reset.status_code == 200
        st_reset = res_reset.json()
        assert st_reset["current_frame"] == 0
        assert st_reset["playback_state"] == "STOPPED"

    def test_snapshot_contains_candidate_obstacles_and_playback_info(self, client):
        """Verify GET /api/map/snapshot includes candidate obstacles and playback metadata."""
        client.post("/api/stream/step", json={"direction": 1})
        res_snap = client.get("/api/map/snapshot")
        assert res_snap.status_code == 200
        snap = res_snap.json()
        assert snap["available"] is True
        assert "candidate_obstacles_count" in snap
        assert "terrain_risk" in snap
        assert "playback" in snap
        assert snap["playback"]["mode"] in ("replay", "synthetic")

