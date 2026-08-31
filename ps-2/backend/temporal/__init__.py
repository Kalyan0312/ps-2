"""
Temporal Processing and Change Detection module initialization.

Provides multi-frame temporal stream processing, state management,
occupancy transition tracking, elevation change detection, and safe incremental
adaptive mapping across 2.5D grid maps and AdaptiveMap25D instances.
"""

from backend.temporal.config import TemporalConfig, TemporalChangeConfig
from backend.temporal.result import (
    TemporalChangeResult,
    STATE_UNOCCUPIED,
    STATE_UNCHANGED,
    STATE_CHANGED,
    STATE_NEWLY_OBSERVED,
    STATE_NO_LONGER_OBSERVED,
)
from backend.temporal.state import TemporalState
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.processor import TemporalProcessor
from backend.temporal.temporal_manager import TemporalMapManager
from backend.temporal.incremental_builder import (
    IncrementalBuildResult,
    IncrementalAdaptiveMapBuilder,
)

__all__ = [
    "TemporalConfig",
    "TemporalChangeConfig",
    "TemporalChangeResult",
    "STATE_UNOCCUPIED",
    "STATE_UNCHANGED",
    "STATE_CHANGED",
    "STATE_NEWLY_OBSERVED",
    "STATE_NO_LONGER_OBSERVED",
    "TemporalState",
    "TemporalChangeDetector",
    "TemporalProcessor",
    "TemporalMapManager",
    "IncrementalBuildResult",
    "IncrementalAdaptiveMapBuilder",
]
