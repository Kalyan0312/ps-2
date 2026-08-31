"""
Temporal Map State Manager.

Coordinates temporal change detection and multi-frame memory across consecutive
AdaptiveMap25D instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.temporal.config import TemporalConfig, TemporalChangeConfig
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.result import TemporalChangeResult


class TemporalMapManager:
    """
    State manager for temporal change detection between consecutive AdaptiveMap25D instances.

    Responsibilities:
      1. Maintains the previous AdaptiveMap25D in memory.
      2. Accepts a new AdaptiveMap25D on each frame update.
      3. For the initial frame:
         - Stores the current map.
         - Returns an initial TemporalChangeResult with `is_initial_frame = True`.
      4. For consecutive frames:
         - Runs TemporalChangeDetector between previous and current maps.
         - Updates the stored previous map.
         - Returns the resulting TemporalChangeResult.

    Args:
        config: TemporalConfig or TemporalChangeConfig instance.
    """

    def __init__(self, config: Optional[Union[TemporalConfig, TemporalChangeConfig]] = None):
        self.config = config or TemporalConfig()
        self.detector = TemporalChangeDetector(config=self.config)
        self.previous_map: Optional[AdaptiveMap25D] = None
        self.last_result: Optional[TemporalChangeResult] = None
        self.frame_count: int = 0
        self.history: List[TemporalChangeResult] = []

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "TemporalMapManager":
        """Instantiates TemporalMapManager directly from YAML configuration file."""
        cfg = TemporalConfig.from_yaml(config_path)
        return cls(config=cfg)

    def reset(self) -> None:
        """Resets temporal tracking state."""
        self.previous_map = None
        self.last_result = None
        self.frame_count = 0
        self.history.clear()

    def update(
        self,
        current_map: AdaptiveMap25D,
        frame_id: Optional[Any] = None,
    ) -> TemporalChangeResult:
        """
        Processes a newly built AdaptiveMap25D against the prior temporal state.

        Args:
            current_map: Current frame's AdaptiveMap25D.
            frame_id: Optional identifier for current frame.

        Returns:
            TemporalChangeResult: Classified change states, masks, and delta stats.
        """
        cid = frame_id if frame_id is not None else current_map.metadata.get("source_frame_id", self.frame_count)

        result = self.detector.detect_change(
            current_grid=current_map,
            previous_grid=self.previous_map,
            frame_index=self.frame_count,
            current_frame_id=cid,
            previous_frame_id=self.previous_map.metadata.get("source_frame_id") if self.previous_map is not None else None,
        )

        self.previous_map = current_map
        self.last_result = result
        self.frame_count += 1
        self.history.append(result)

        return result

    @property
    def has_history(self) -> bool:
        """True if at least one map has been processed and stored."""
        return self.previous_map is not None and self.frame_count > 0

    @property
    def last_map(self) -> Optional[AdaptiveMap25D]:
        """Returns the most recent AdaptiveMap25D stored in state."""
        return self.previous_map

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of temporal tracking state."""
        return {
            "frame_count": self.frame_count,
            "has_history": self.has_history,
            "history_length": len(self.history),
            "last_frame_id": self.last_result.current_frame_id if self.last_result is not None else None,
        }
