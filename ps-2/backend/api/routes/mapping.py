"""
Spatial Map Query Endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.session import session_manager

router = APIRouter(prefix="/api/map", tags=["Mapping"])


class MapQueryRequest(BaseModel):
    x: float = Field(..., description="World X coordinate in meters")
    y: float = Field(..., description="World Y coordinate in meters")


@router.post("/query")
async def query_adaptive_map(req: MapQueryRequest):
    """
    Performs constant-time O(1) spatial query on the latest processed AdaptiveMap25D.
    """
    return session_manager.query_map(x=req.x, y=req.y)


@router.get("/snapshot")
async def get_adaptive_map_snapshot():
    """
    Returns live JSON snapshot of the latest processed AdaptiveMap25D.
    """
    return session_manager.get_map_snapshot()

