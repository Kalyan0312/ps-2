import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

from backend.api.main import app
from backend.api.session import StreamSessionManager

@pytest.fixture
def test_client():
    return TestClient(app)

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@patch("backend.api.routes.replay.session_manager")
def test_replay_start_endpoint(mock_session_manager, test_client):
    mock_session_manager.start_stream = AsyncMock(return_value={
        "status": "started",
        "playback_state": "PLAYING",
    })
    
    response = test_client.post("/api/replay/start", json={"fps": 10.0, "speed": 1.0, "mode": "replay"})
    assert response.status_code == 200
    assert response.json()["status"] == "started"

@patch("backend.api.routes.replay.session_manager")
def test_replay_pause_endpoint(mock_session_manager, test_client):
    mock_session_manager.pause_stream = AsyncMock(return_value={
        "status": "paused",
        "playback_state": "PAUSED",
    })
    
    response = test_client.post("/api/replay/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"

@patch("backend.api.routes.replay.session_manager")
def test_replay_reset_endpoint(mock_session_manager, test_client):
    mock_session_manager.reset = MagicMock(return_value={"playback_state": "STOPPED", "current_frame": 0})
    
    response = test_client.post("/api/replay/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["playback_state"] == "STOPPED"

@patch("backend.api.routes.replay.session_manager")
def test_replay_step_endpoint(mock_session_manager, test_client):
    mock_session_manager.step_frame = MagicMock(return_value={"current_frame": 1})
    
    response = test_client.post("/api/replay/step?direction=1")
    assert response.status_code == 200
    assert response.json()["current_frame"] == 1

@patch("backend.api.routes.replay.session_manager")
def test_replay_status_endpoint(mock_session_manager, test_client):
    mock_session_manager.get_status = MagicMock(return_value={
        "is_streaming": False,
        "playback_state": "STOPPED",
        "current_frame": 0,
        "total_frames": 5,
    })
    
    response = test_client.get("/api/replay/status")
    assert response.status_code == 200
    data = response.json()
    assert data["playback_state"] == "STOPPED"
    assert data["total_frames"] == 5
