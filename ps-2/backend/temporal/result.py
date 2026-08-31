"""
Temporal Change Detection Result Data Structure.

Encapsulates spatial masks, change state matrices, change statistics,
and elevation deltas between consecutive 2.5D grid maps or AdaptiveMap25D instances.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np


# Change State String Constants
STATE_UNOCCUPIED = "unoccupied"
STATE_UNCHANGED = "unchanged"
STATE_CHANGED = "changed"
STATE_NEWLY_OBSERVED = "newly_observed"
STATE_NO_LONGER_OBSERVED = "no_longer_observed"


class TemporalChangeResult:
    """
    Result of temporal change analysis between two consecutive 2.5D elevation grid maps
    or AdaptiveMap25D instances.

    State representation in `change_state`:
      - "unchanged": Cell is represented in both maps with |delta_z| < threshold.
      - "changed": Cell is represented in both maps with |delta_z| >= threshold.
      - "newly_observed": Cell is represented in current map but was not in previous map.
      - "no_longer_observed": Cell was represented in previous map but is not in current map.
      - "unoccupied": Cell is unrepresented/empty in both maps.
    """

    def __init__(
        self,
        frame_index: int,
        previous_frame_id: Optional[Any],
        current_frame_id: Any,
        grid_shape: Tuple[int, int],
        resolution: float,
        bounds: Tuple[float, float, float, float],
        is_initial_frame: bool,
        change_state: Optional[np.ndarray] = None,
        elevation_difference: Optional[np.ndarray] = None,
        changed_mask: Optional[np.ndarray] = None,
        unchanged_mask: Optional[np.ndarray] = None,
        newly_observed_mask: Optional[np.ndarray] = None,
        no_longer_observed_mask: Optional[np.ndarray] = None,
        unoccupied_mask: Optional[np.ndarray] = None,
        elevation_changed_mask: Optional[np.ndarray] = None,
        # Backward-compatible keyword arguments from Phase 9:
        new_occupancy_mask: Optional[np.ndarray] = None,
        removed_occupancy_mask: Optional[np.ndarray] = None,
        stable_mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.frame_index = frame_index
        self.previous_frame_id = previous_frame_id
        self.current_frame_id = current_frame_id
        self.grid_shape = grid_shape
        self.resolution = resolution
        self.bounds = bounds
        self.is_initial_frame = is_initial_frame
        self.metadata = metadata or {}

        # Resolve newly_observed_mask / new_occupancy_mask
        if newly_observed_mask is not None:
            self.newly_observed_mask = newly_observed_mask
        elif new_occupancy_mask is not None:
            self.newly_observed_mask = new_occupancy_mask
        else:
            self.newly_observed_mask = np.zeros(grid_shape, dtype=bool)

        # Resolve no_longer_observed_mask / removed_occupancy_mask
        if no_longer_observed_mask is not None:
            self.no_longer_observed_mask = no_longer_observed_mask
        elif removed_occupancy_mask is not None:
            self.no_longer_observed_mask = removed_occupancy_mask
        else:
            self.no_longer_observed_mask = np.zeros(grid_shape, dtype=bool)

        # Resolve unchanged_mask / stable_mask
        if unchanged_mask is not None:
            self.unchanged_mask = unchanged_mask
        elif stable_mask is not None:
            self.unchanged_mask = stable_mask
        else:
            self.unchanged_mask = np.zeros(grid_shape, dtype=bool)

        # Elevation changed mask
        if elevation_changed_mask is not None:
            self.elevation_changed_mask = elevation_changed_mask
        else:
            self.elevation_changed_mask = np.zeros(grid_shape, dtype=bool)

        # Changed mask
        if changed_mask is not None:
            self.changed_mask = changed_mask
        else:
            self.changed_mask = (
                self.newly_observed_mask |
                self.no_longer_observed_mask |
                self.elevation_changed_mask
            )

        # Unoccupied mask
        if unoccupied_mask is not None:
            self.unoccupied_mask = unoccupied_mask
        else:
            self.unoccupied_mask = ~(
                self.unchanged_mask |
                self.changed_mask
            )

        # Elevation difference
        if elevation_difference is not None:
            self.elevation_difference = elevation_difference
        else:
            self.elevation_difference = np.full(grid_shape, np.nan, dtype=np.float32)

        # Change state matrix
        if change_state is not None:
            self.change_state = change_state
        else:
            self.change_state = np.full(grid_shape, STATE_UNOCCUPIED, dtype=object)
            self.change_state[self.unchanged_mask] = STATE_UNCHANGED
            self.change_state[self.elevation_changed_mask] = STATE_CHANGED
            self.change_state[self.newly_observed_mask] = STATE_NEWLY_OBSERVED
            self.change_state[self.no_longer_observed_mask] = STATE_NO_LONGER_OBSERVED

        self._validate_shapes()

    def _validate_shapes(self):
        expected_shape = self.grid_shape
        for name, arr in [
            ("change_state", self.change_state),
            ("elevation_difference", self.elevation_difference),
            ("changed_mask", self.changed_mask),
            ("unchanged_mask", self.unchanged_mask),
            ("newly_observed_mask", self.newly_observed_mask),
            ("no_longer_observed_mask", self.no_longer_observed_mask),
            ("unoccupied_mask", self.unoccupied_mask),
            ("elevation_changed_mask", self.elevation_changed_mask),
        ]:
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Layer '{name}' shape {arr.shape} does not match expected grid shape {expected_shape}"
                )

    # ------------------------------------------------------------------
    # Backward Compatibility Aliases for Phase 9
    # ------------------------------------------------------------------
    @property
    def new_occupancy_mask(self) -> np.ndarray:
        """Alias for newly_observed_mask."""
        return self.newly_observed_mask

    @property
    def removed_occupancy_mask(self) -> np.ndarray:
        """Alias for no_longer_observed_mask."""
        return self.no_longer_observed_mask

    @property
    def stable_mask(self) -> np.ndarray:
        """Alias for unchanged_mask."""
        return self.unchanged_mask

    # ------------------------------------------------------------------
    # Dimensions & Counts
    # ------------------------------------------------------------------
    @property
    def rows(self) -> int:
        return self.grid_shape[0]

    @property
    def cols(self) -> int:
        return self.grid_shape[1]

    @property
    def total_grid_cells(self) -> int:
        """Total grid cells in the spatial domain (rows * cols)."""
        return self.rows * self.cols

    @property
    def total_compared_cells(self) -> int:
        """Total count of cells represented in both maps and directly compared."""
        return int(np.count_nonzero(self.unchanged_mask | self.elevation_changed_mask))

    @property
    def changed_cells(self) -> int:
        """Count of all changed cells (newly observed + no longer observed + elevation changed)."""
        return int(np.count_nonzero(self.changed_mask))

    @property
    def unchanged_cells(self) -> int:
        """Count of persistent cells with elevation delta below threshold."""
        return int(np.count_nonzero(self.unchanged_mask))

    @property
    def newly_observed_cells(self) -> int:
        """Count of cells newly observed in the current frame."""
        return int(np.count_nonzero(self.newly_observed_mask))

    @property
    def no_longer_observed_cells(self) -> int:
        """Count of cells present previously but absent in current frame."""
        return int(np.count_nonzero(self.no_longer_observed_mask))

    @property
    def num_changed_cells(self) -> int:
        return self.changed_cells

    @property
    def num_stable_cells(self) -> int:
        return self.unchanged_cells

    @property
    def num_new_occupied_cells(self) -> int:
        return self.newly_observed_cells

    @property
    def num_removed_occupied_cells(self) -> int:
        return self.no_longer_observed_cells

    @property
    def num_elevation_changed_cells(self) -> int:
        return int(np.count_nonzero(self.elevation_changed_mask))

    @property
    def num_unoccupied_cells(self) -> int:
        return int(np.count_nonzero(self.unoccupied_mask))

    @property
    def total_active_cells(self) -> int:
        """Total cells containing points in either previous or current frame."""
        return self.changed_cells + self.unchanged_cells

    @property
    def change_percentage(self) -> float:
        """Percentage of active cells that changed between frames [0.0%, 100.0%]."""
        if self.total_active_cells == 0 or self.is_initial_frame:
            return 0.0
        return (self.changed_cells / self.total_active_cells) * 100.0

    @property
    def stable_percentage(self) -> float:
        """Percentage of active cells that remained stable [0.0%, 100.0%]."""
        if self.total_active_cells == 0:
            return 100.0
        if self.is_initial_frame:
            return 100.0
        return (self.unchanged_cells / self.total_active_cells) * 100.0

    # ------------------------------------------------------------------
    # Query & Summaries
    # ------------------------------------------------------------------
    def get_cell(self, row: int, col: int) -> Dict[str, Any]:
        """Returns temporal status dictionary for a specific grid cell."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"Cell ({row}, {col}) out of grid bounds {self.grid_shape}")

        state = str(self.change_state[row, col])
        is_new = bool(self.newly_observed_mask[row, col])
        is_rem = bool(self.no_longer_observed_mask[row, col])
        is_elev = bool(self.elevation_changed_mask[row, col])
        is_chg = bool(self.changed_mask[row, col])
        is_stb = bool(self.unchanged_mask[row, col])
        delta_z = float(self.elevation_difference[row, col]) if not np.isnan(self.elevation_difference[row, col]) else None

        # Backwards compatible status string for Phase 9 assertions
        status = state
        if status == STATE_UNCHANGED:
            status = "stable"
        elif status == STATE_NEWLY_OBSERVED:
            status = "new_occupancy"
        elif status == STATE_NO_LONGER_OBSERVED:
            status = "removed_occupancy"
        elif status == STATE_CHANGED:
            status = "elevation_changed"

        return {
            "row": row,
            "col": col,
            "state": state,
            "status": status,
            "is_changed": is_chg,
            "is_stable": is_stb,
            "is_unchanged": is_stb,
            "is_new_occupancy": is_new,
            "is_newly_observed": is_new,
            "is_removed_occupancy": is_rem,
            "is_no_longer_observed": is_rem,
            "is_elevation_changed": is_elev,
            "elevation_difference": delta_z,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured diagnostic summary of temporal changes."""
        valid_diffs = self.elevation_difference[~np.isnan(self.elevation_difference)]
        max_diff = float(np.max(np.abs(valid_diffs))) if len(valid_diffs) > 0 else 0.0
        mean_diff = float(np.mean(np.abs(valid_diffs))) if len(valid_diffs) > 0 else 0.0

        return {
            "frame_index": self.frame_index,
            "previous_frame_id": self.previous_frame_id,
            "current_frame_id": self.current_frame_id,
            "is_initial_frame": self.is_initial_frame,
            "total_grid_cells": self.total_grid_cells,
            "total_compared_cells": self.total_compared_cells,
            "total_active_cells": self.total_active_cells,
            "changed_cells": self.changed_cells,
            "unchanged_cells": self.unchanged_cells,
            "newly_observed_cells": self.newly_observed_cells,
            "no_longer_observed_cells": self.no_longer_observed_cells,
            "elevation_changed_cells": self.num_elevation_changed_cells,
            "max_elevation_difference": round(max_diff, 4),
            "mean_elevation_difference": round(mean_diff, 4),
            "change_percentage": round(self.change_percentage, 2),
            "stable_percentage": round(self.stable_percentage, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_summary()
