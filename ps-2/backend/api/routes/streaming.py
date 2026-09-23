"""
Streaming Session REST Endpoints.
"""

from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.api.session import session_manager

router = APIRouter(prefix="/api/stream", tags=["Streaming"])


class StreamStartRequest(BaseModel):
    fps: float = Field(default=15.0, ge=1.0, le=60.0, description="Target streaming frame rate")
    total_frames: Optional[int] = Field(default=None, ge=1, description="Optional maximum frames to process")
    speed: float = Field(default=1.0, ge=0.1, le=10.0, description="Playback speed multiplier")
    mode: Optional[str] = Field(default="replay", description="Playback mode: 'replay' or 'synthetic'")


class StepRequest(BaseModel):
    direction: int = Field(default=1, description="Frame step direction: +1 (next) or -1 (previous)")


class SeekRequest(BaseModel):
    frame_idx: int = Field(default=0, ge=0, description="Target frame index to seek")


class SpeedRequest(BaseModel):
    speed: float = Field(default=1.0, ge=0.1, le=10.0, description="Playback speed multiplier (0.5x, 1x, 2x, 4x)")


@router.post("/start")
async def start_stream(req: Optional[StreamStartRequest] = None):
    """
    Starts LiDAR streaming session or dataset replay.
    """
    fps = req.fps if req else 15.0
    total_frames = req.total_frames if req else None
    speed = req.speed if req else 1.0
    mode = req.mode if req else "replay"
    return await session_manager.start_stream(fps=fps, total_frames=total_frames, speed=speed, mode=mode)


@router.post("/play")
async def play_stream(req: Optional[StreamStartRequest] = None):
    """
    Starts or resumes LiDAR dataset playback.
    """
    return await start_stream(req)


@router.post("/stop")
async def stop_stream():
    """
    Stops or pauses active LiDAR playback session.
    """
    return await session_manager.stop_stream()


@router.post("/pause")
async def pause_stream():
    """
    Pauses active LiDAR playback session.
    """
    return await session_manager.pause_stream()


@router.post("/reset")
async def reset_stream():
    """
    Resets playback playhead index to 0 and clears map state.
    """
    return session_manager.reset()


@router.post("/step")
async def step_stream(req: Optional[StepRequest] = None, direction: int = Query(1)):
    """
    Steps forward (+1) or backward (-1) by one frame in the recorded dataset.
    """
    step_dir = req.direction if req else direction
    return session_manager.step_frame(direction=step_dir)


@router.post("/seek")
async def seek_stream(req: Optional[SeekRequest] = None, frame_idx: int = Query(0)):
    """
    Seeks directly to frame index in the recorded dataset.
    """
    target = req.frame_idx if req else frame_idx
    return session_manager.seek_frame(frame_idx=target)


@router.post("/speed")
async def set_speed(req: Optional[SpeedRequest] = None, speed: float = Query(1.0)):
    """
    Sets playback speed multiplier (0.5, 1.0, 2.0, 4.0).
    """
    sp = req.speed if req else speed
    return session_manager.set_speed(speed=sp)


@router.get("/status")
async def get_stream_status():
    """
    Returns current streaming pipeline metrics, throughput, playback position, and state.
    """
    return session_manager.get_status()

