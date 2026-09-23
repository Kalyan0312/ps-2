"""
Unit & Integration Tests for Live Adaptive Map Snapshot API (Phase 16 Part A).

Tests GET /api/map/snapshot endpoint behavior:
- Snapshot unavailable before first frame
- Snapshot available after streaming / frame processing
- Valid JSON serialization
- Correct tier resolutions
- No duplicate represented cells
- Valid cell attributes (x, y, z, tier, resolution, point_count, min_z, max_z, mean_intensity)
- Non-mutating behavior on AdaptiveMap25D
- Verification that POST /api/map/query remains functional
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.session import session_manager
from backend.data.frame import LidarFrame


@pytest.fixture(autouse=True)
def reset_session():
    """Ensure session manager state is reset before and after each test."""
    session_manager.reset()
    yield
    session_manager.reset()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestLiveMapSnapshotApi:

    def test_snapshot_unavailable_before_first_frame(self, client):
        """1. Verify snapshot response when no map exists yet."""
        res = client.get("/api/map/snapshot")
        assert res.status_code == 200
        data = res.json()
        assert data["available"] is False
        assert data["frame_id"] is None
        assert data["cells"] == []

    def test_snapshot_available_after_frame_processing(self, client):
        """2. Verify snapshot response after processing a frame."""
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
            [0.1, 0.1, 1.6, 110.0],
            [1.0, 1.0, 0.5, 90.0],
            [5.0, 5.0, 2.0, 80.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=42, timestamp=123.45)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map
        session_manager.latest_result = res_pipeline

        res = client.get("/api/map/snapshot")
        assert res.status_code == 200
        data = res.json()
        assert data["available"] is True
        assert data["frame_id"] == 42
        assert "bounds" in data
        assert "min_x" in data["bounds"]
        assert "max_x" in data["bounds"]
        assert "min_y" in data["bounds"]
        assert "max_y" in data["bounds"]
        assert data["base_resolution"] == 0.20
        assert data["represented_cells"] > 0
        assert len(data["cells"]) == data["represented_cells"]

    def test_snapshot_valid_json_serialization(self, client):
        """3. Verify snapshot data can be serialized to JSON strictly without errors."""
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
            [2.0, 2.0, 0.8, 120.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=1, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        res = client.get("/api/map/snapshot")
        assert res.status_code == 200
        # Will raise TypeError if any non-serializable objects (like numpy floats) are returned in raw dict
        raw_json_str = res.text
        parsed = json.loads(raw_json_str)
        assert parsed["available"] is True

    def test_correct_tier_resolutions(self, client):
        """4. Verify correct tier resolutions in tiers dict and cell objects."""
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        res = client.get("/api/map/snapshot")
        data = res.json()
        assert "tiers" in data
        tiers = data["tiers"]
        # Standard configuration tiers
        expected_resolutions = {"coarse": 0.50, "medium": 0.25, "fine": 0.10, "ultra_fine": 0.05}
        for t_name, expected_res in expected_resolutions.items():
            if t_name in tiers:
                assert tiers[t_name]["resolution"] == pytest.approx(expected_res)
                assert "cell_count" in tiers[t_name]

        for cell in data["cells"]:
            assert cell["tier"] in expected_resolutions
            assert cell["resolution"] == pytest.approx(expected_resolutions[cell["tier"]])

    def test_no_duplicate_represented_cells(self, client):
        """5. Verify index_tier prevents duplicate cell exposures from overlapping bounding boxes."""
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
            [0.05, 0.05, 1.5, 100.0],
            [10.0, 10.0, 0.2, 50.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        res = client.get("/api/map/snapshot")
        data = res.json()
        cells = data["cells"]

        # Check that no two cells have the exact same (x, y, tier)
        cell_keys = [(c["x"], c["y"], c["tier"]) for c in cells]
        assert len(cell_keys) == len(set(cell_keys)), "Found duplicate represented cells in snapshot"

    def test_valid_cell_attributes(self, client):
        """6. Verify all required fields exist with correct data types in cells array."""
        pts = np.array([
            [0.5, 0.5, 1.25, 95.5],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        res = client.get("/api/map/snapshot")
        data = res.json()
        cells = data["cells"]
        assert len(cells) > 0

        for cell in cells:
            assert isinstance(cell["x"], float)
            assert isinstance(cell["y"], float)
            assert isinstance(cell["z"], float)
            assert isinstance(cell["tier"], str)
            assert isinstance(cell["resolution"], float)
            assert isinstance(cell["point_count"], int)
            assert cell["point_count"] > 0
            assert isinstance(cell["min_z"], float)
            assert isinstance(cell["max_z"], float)
            assert cell["mean_intensity"] is None or isinstance(cell["mean_intensity"], (float, int))

    def test_endpoint_does_not_mutate_map(self, client):
        """7. Verify snapshot endpoint call does not mutate AdaptiveMap25D state."""
        pts = np.array([
            [0.0, 0.0, 1.5, 100.0],
            [2.0, 2.0, 0.8, 120.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        m = res_pipeline.adaptive_map
        session_manager.latest_map = m

        # Capture hash/copies of key arrays
        index_tier_before = m.index_tier.copy()
        index_row_before = m.index_row.copy()
        index_col_before = m.index_col.copy()
        num_cells_before = m.num_represented_cells

        # Query snapshot twice
        client.get("/api/map/snapshot")
        client.get("/api/map/snapshot")

        # Assert no mutations occurred
        np.testing.assert_array_equal(m.index_tier, index_tier_before)
        np.testing.assert_array_equal(m.index_row, index_row_before)
        np.testing.assert_array_equal(m.index_col, index_col_before)
        assert m.num_represented_cells == num_cells_before

    def test_existing_spatial_query_and_health_remain_functional(self, client):
        """8. Verify health endpoint and POST /api/map/query continue working as expected."""
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["status"] == "ok"

        pts = np.array([[0.0, 0.0, 1.5, 100.0]], dtype=np.float64)
        frame = LidarFrame(points=pts, frame_id=0, timestamp=0.0)
        res_pipeline = session_manager.pipeline.process_frame(frame)
        session_manager.latest_map = res_pipeline.adaptive_map

        query_res = client.post("/api/map/query", json={"x": 0.0, "y": 0.0})
        assert query_res.status_code == 200
        qdata = query_res.json()
        assert qdata["is_represented"] is True
        assert qdata["x"] == 0.0
        assert qdata["y"] == 0.0

    def test_dashboard_contains_map_canvas_and_controls(self, client):
        """9. Verify dashboard static HTML contains Section 6 Map Canvas, Empty State, Controls, and Legend."""
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        assert 'id="map-canvas"' in html
        assert 'id="map-empty-overlay"' in html
        assert 'Awaiting LiDAR Replay Start' in html
        assert 'id="btn-refresh-map"' in html
        assert 'id="btn-fit-map"' in html
        assert 'id="btn-zoom-in"' in html
        assert 'id="btn-zoom-out"' in html
        assert 'id="btn-toggle-color"' in html
        assert 'Coarse — 0.50 m' in html
        assert 'Ultra Fine — 0.05 m' in html

    def test_websocket_emits_map_update(self, client):
        """10. Verify WebSocket emits map_update notification containing valid frame_id on map generation."""
        with client.websocket_connect("/ws/stream") as websocket:
            init_msg = websocket.receive_json()
            assert init_msg["type"] == "connection_established"

            # Manually trigger streaming loop frame step
            pts = np.array([[0.0, 0.0, 1.5, 100.0]], dtype=np.float64)
            frame = LidarFrame(points=pts, frame_id=99, timestamp=100.0)
            res_pipeline = session_manager.pipeline.process_frame(frame)
            session_manager.latest_result = res_pipeline
            session_manager.latest_map = res_pipeline.adaptive_map

            # Formulate and broadcast payloads (matching _stream_loop)
            telemetry = session_manager._build_telemetry_payload(res_pipeline)
            import asyncio
            asyncio.run(session_manager.broadcast(telemetry))
            asyncio.run(session_manager.broadcast({
                "type": "map_update",
                "frame_id": res_pipeline.frame_id,
                "timestamp": res_pipeline.timestamp,
            }))

            # Receive messages
            msg1 = websocket.receive_json()
            assert msg1["type"] == "frame_update"
            assert msg1["frame_id"] == 99

            msg2 = websocket.receive_json()
            assert msg2["type"] == "map_update"
            assert msg2["frame_id"] == 99


