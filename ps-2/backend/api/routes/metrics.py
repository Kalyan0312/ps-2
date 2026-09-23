"""
System and Pipeline Metrics Endpoints.
"""

from fastapi import APIRouter
from backend.api.session import session_manager

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])


@router.get("")
async def get_metrics():
    """
    Returns comprehensive pipeline performance metrics, strategy history, and map diagnostics.
    """
    metrics = session_manager.pipeline.get_performance_metrics()
    
    map_summary = {}
    if session_manager.latest_map is not None:
        map_summary = {
            "num_represented_cells": session_manager.latest_map.num_represented_cells,
            "base_shape": session_manager.latest_map.base_shape,
            "base_resolution": session_manager.latest_map.base_resolution,
            "tier_resolutions": session_manager.latest_map.tier_resolutions,
            "available_tiers": list(session_manager.latest_map.tier_maps.keys()),
        }

    return {
        "pipeline_metrics": metrics,
        "current_fps": session_manager.current_fps,
        "is_streaming": session_manager.is_streaming,
        "has_active_map": session_manager.latest_map is not None,
        "active_map_summary": map_summary,
    }
