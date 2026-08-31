"""
Temporal State Manager.

Maintains multi-frame temporal context, persistent stability tracking, and
change history across consecutive LiDAR frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.temporal.result import TemporalChangeResult


@dataclass
class TemporalState:
    """
    State container maintaining temporal memory across sequential LiDAR frames.

    Attributes:
        last_grid_map: GridMap25D from the most recently processed frame.
        last_frame_id: Identifier of the most recent frame.
        last_result: Most recent TemporalChangeResult.
        frame_count: Number of frames ingested in the current sequence.
        history: Chronological list of all TemporalChangeResult objects.
        persistent_stable_frames: 2D int32 array tracking consecutive stable frame counts per cell.
    """
    last_grid_map: Optional[GridMap25D] = None
    last_frame_id: Optional[Any] = None
    last_result: Optional[TemporalChangeResult] = None
    frame_count: int = 0
    history: List[TemporalChangeResult] = field(default_factory=list)
    persistent_stable_frames: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets temporal memory to clean initial state."""
        self.last_grid_map = None
        self.last_frame_id = None
        self.last_result = None
        self.frame_count = 0
        self.history.clear()
        self.persistent_stable_frames = None

    def update(self, grid_map: GridMap25D, result: TemporalChangeResult) -> None:
        """
        Updates temporal state with the newly processed grid map and change detection result.
        """
        rows, cols = grid_map.shape
        if self.persistent_stable_frames is None or self.persistent_stable_frames.shape != (rows, cols):
            self.persistent_stable_frames = np.zeros((rows, cols), dtype=np.int32)

        if result.is_initial_frame:
            # First frame: initialize stable counts for occupied cells
            self.persistent_stable_frames[result.stable_mask] = 1
            self.persistent_stable_frames[~result.stable_mask] = 0
        else:
            # Consecutive frames: increment stable cells, reset changed/empty cells
            self.persistent_stable_frames[result.stable_mask] += 1
            self.persistent_stable_frames[~result.stable_mask] = 0

        self.last_grid_map = grid_map
        self.last_frame_id = result.current_frame_id
        self.last_result = result
        self.frame_count += 1
        self.history.append(result)

    @property
    def has_history(self) -> bool:
        """True if at least one frame has been processed."""
        return self.last_grid_map is not None and self.frame_count > 0

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of temporal tracking state."""
        max_stable_run = int(np.max(self.persistent_stable_frames)) if self.persistent_stable_frames is not None else 0
        return {
            "frame_count": self.frame_count,
            "last_frame_id": self.last_frame_id,
            "history_length": len(self.history),
            "max_consecutive_stable_frames": max_stable_run,
        }
