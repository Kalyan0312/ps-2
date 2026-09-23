"""
LiDAR Dataset Replay REST Endpoints.

Provides single-process playback controls for recorded LiDAR datasets:
start, pause, reset, step, seek, speed, and status.
"""

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.api.session import session_manager

router = APIRouter(prefix="/api/replay", tags=["Replay"])


class ReplayStartRequest(BaseModel):
    fps: float = Field(default=10.0, ge=1.0, le=60.0, description="Target replay frame rate")
    total_frames: Optional[int] = Field(default=None, ge=1, description="Optional maximum frames to replay")
    speed: float = Field(default=0.25, ge=0.1, le=10.0, description="Playback speed multiplier")
    mode: Optional[str] = Field(default="replay", description="Playback mode: 'replay' or 'synthetic'")


class ReplayStepRequest(BaseModel):
    direction: int = Field(default=1, description="Frame step direction: +1 (forward) or -1 (backward)")


class ReplaySeekRequest(BaseModel):
    frame_idx: int = Field(default=0, ge=0, description="Target frame index to seek")


class ReplaySpeedRequest(BaseModel):
    speed: float = Field(default=0.25, ge=0.1, le=10.0, description="Playback speed multiplier (0.5x, 1x, 2x, 4x)")


@router.post("/start")
async def start_replay(req: Optional[ReplayStartRequest] = None):
    """
    Starts or resumes recorded LiDAR dataset replay inside the FastAPI process.
    """
    fps = req.fps if req else 10.0
    total_frames = req.total_frames if req else None
    speed = req.speed if req else 1.0
    mode = req.mode if req else "replay"
    return await session_manager.start_stream(fps=fps, total_frames=total_frames, speed=speed, mode=mode)


@router.post("/play")
async def play_replay(req: Optional[ReplayStartRequest] = None):
    """
    Alias for start: starts or resumes playback.
    """
    return await start_replay(req)


@router.post("/pause")
async def pause_replay():
    """
    Pauses active LiDAR dataset replay.
    """
    return await session_manager.pause_stream()


@router.post("/stop")
async def stop_replay():
    """
    Stops active LiDAR dataset replay.
    """
    return await session_manager.stop_stream()


@router.post("/reset")
async def reset_replay():
    """
    Resets playhead back to frame 0 and resets pipeline state.
    """
    return session_manager.reset()


@router.post("/step")
async def step_replay(req: Optional[ReplayStepRequest] = None, direction: int = Query(1)):
    """
    Steps forward (+1) or backward (-1) by one frame in the dataset.
    """
    step_dir = req.direction if req else direction
    return session_manager.step_frame(direction=step_dir)


@router.post("/seek")
async def seek_replay(req: Optional[ReplaySeekRequest] = None, frame_idx: int = Query(0)):
    """
    Seeks directly to a specific frame index in the recorded dataset.
    """
    target = req.frame_idx if req else frame_idx
    return session_manager.seek_frame(frame_idx=target)


@router.post("/speed")
async def set_replay_speed(req: Optional[ReplaySpeedRequest] = None, speed: float = Query(0.25)):
    """
    Sets playback speed multiplier (0.5, 1.0, 2.0, 4.0).
    """
    sp = req.speed if req else speed
    return session_manager.set_speed(speed=sp)


@router.get("/status")
async def get_replay_status():
    """
    Returns current replay position, frame counts, playback state, latency, and throughput.
    """
    return session_manager.get_status()
