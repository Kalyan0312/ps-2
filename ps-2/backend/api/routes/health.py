"""
Health Check Endpoint.
"""

from fastapi import APIRouter
from backend.api.session import session_manager

router = APIRouter(tags=["Health"])


@router.get("/health")
async def get_health():
    """
    Returns system health, pipeline readiness, and active streaming state.
    """
    return {
        "status": "ok",
        "service": "Adaptive LiDAR 2.5D Mapping System",
        "pipeline_ready": session_manager.pipeline is not None,
        "is_streaming": session_manager.is_streaming,
        "has_active_map": session_manager.latest_map is not None,
        "active_clients": len(session_manager._active_clients),
    }
