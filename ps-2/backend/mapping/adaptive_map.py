"""
Adaptive 2.5D Elevation Map Data Structure.

Represents a composite spatial map where each occupied region is stored at
the resolution tier assigned by the AdaptiveResolutionSelector (Phase 5).

Architecture
------------
The map is built from a ResolutionDecisionResult whose `assigned_level` matrix
indicates, for every base-grid cell, which resolution tier applies.

For each tier the builder constructs one GridMap25D at that tier's resolution,
covering only the spatial extents where that tier was assigned.  The
AdaptiveMap25D stores:

  - The four per-tier GridMap25D instances (only tiers that have ≥1 assigned
    cell are populated).
  - A lightweight index matrix (same shape as the base grid) that maps every
    base cell → (tier name, fine-grid row, fine-grid col).  This makes
    coordinate queries O(1) after construction.

Querying
--------
world_query(x, y) → AdaptiveCellInfo
  Returns the tier, numerical resolution, and elevation statistics for any
  world-coordinate pair.

Summary
-------
get_summary() returns counts / percentages per tier and a comparison with a
hypothetical uniform ultra-fine map.

Design is intentionally open for later extension:
  - route-aware priority overrides
  - semantic refinement
  - temporal patch updates
  - compute-budget trimming
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.priority.result import ResolutionDecisionResult


# ---------------------------------------------------------------------------
# Public result container for a single adaptive-map cell query
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveCellInfo:
    """
    Result returned by AdaptiveMap25D.world_query().

    Attributes:
        query_x: World X coordinate that was queried (meters).
        query_y: World Y coordinate that was queried (meters).
        is_represented: True when the location falls inside the adaptive map
            bounds and has at least one LiDAR point.
        tier: Resolution tier name ('coarse', 'medium', 'fine', 'ultra_fine')
            or None if not represented.
        resolution: Numerical cell resolution in meters, or None.
        min_z: Minimum elevation in the tier cell (meters), or None.
        max_z: Maximum elevation in the tier cell (meters), or None.
        mean_z: Mean elevation in the tier cell (meters), or None.
        point_count: Number of LiDAR points aggregated in the tier cell, or None.
        mean_intensity: Mean LiDAR intensity of the tier cell, or None.
    """
    query_x: float
    query_y: float
    is_represented: bool
    tier: Optional[str] = None
    resolution: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None
    mean_z: Optional[float] = None
    point_count: Optional[int] = None
    mean_intensity: Optional[float] = None


# ---------------------------------------------------------------------------
# Core adaptive map data structure
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveMap25D:
    """
    Composite adaptive-resolution 2.5D elevation map.

    Each spatial region is stored at the resolution tier assigned by
    the AdaptiveResolutionSelector.

    Attributes:
        tier_maps: Per-tier GridMap25D instances keyed by tier name.
            Only tiers with ≥1 assigned cell are present.
        tier_resolutions: Tier name → resolution in meters.
        base_shape: (rows, cols) of the reference base grid.
        base_resolution: Resolution of the reference base grid (meters).
        base_bounds: (min_x, max_x, min_y, max_y) of the reference grid.
        index_tier: [rows, cols] object array of tier name strings.
            Empty string '' for unassigned/empty base cells.
        index_row: [rows, cols] int32 array of fine-grid row index for each
            base cell (-1 for unassigned cells).
        index_col: [rows, cols] int32 array of fine-grid column index for each
            base cell (-1 for unassigned cells).
        metadata: Construction metadata.
    """
    tier_maps: Dict[str, GridMap25D]
    tier_resolutions: Dict[str, float]
    base_shape: Tuple[int, int]
    base_resolution: float
    base_bounds: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    index_tier: np.ndarray   # dtype=object, shape=base_shape
    index_row: np.ndarray    # dtype=int32,  shape=base_shape
    index_col: np.ndarray    # dtype=int32,  shape=base_shape
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def min_x(self) -> float:
        return self.base_bounds[0]

    @property
    def max_x(self) -> float:
        return self.base_bounds[1]

    @property
    def min_y(self) -> float:
        return self.base_bounds[2]

    @property
    def max_y(self) -> float:
        return self.base_bounds[3]

    @property
    def available_tiers(self) -> List[str]:
        """Tier names that have ≥1 assigned adaptive cell."""
        return list(self.tier_maps.keys())

    @property
    def num_represented_cells(self) -> int:
        """Total number of base-grid cells with a valid tier assignment."""
        return int(np.count_nonzero(self.index_tier != ""))

    @property
    def tier_cell_counts(self) -> Dict[str, int]:
        """Number of base-grid cells assigned to each tier."""
        return {
            tier: int(np.count_nonzero(self.index_tier == tier))
            for tier in self.tier_resolutions
        }

    @property
    def tier_percentages(self) -> Dict[str, float]:
        """Percentage of represented cells per tier."""
        total = self.num_represented_cells
        if total == 0:
            return {t: 0.0 for t in self.tier_resolutions}
        return {
            tier: (count / total) * 100.0
            for tier, count in self.tier_cell_counts.items()
        }

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def in_bounds(self, x: float, y: float) -> bool:
        """Returns True when (x, y) falls within the map boundaries (inclusive)."""
        return bool(
            self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
        )

    def _base_grid_index(self, x: float, y: float) -> Tuple[int, int]:
        """Maps world (x, y) to the base-grid (row, col)."""
        col = int(np.floor((x - self.min_x) / self.base_resolution))
        row = int(np.floor((y - self.min_y) / self.base_resolution))
        rows, cols = self.base_shape
        col = max(0, min(col, cols - 1))
        row = max(0, min(row, rows - 1))
        return row, col

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def world_query(self, x: float, y: float) -> AdaptiveCellInfo:
        """
        Queries the adaptive map at world coordinates (x, y).

        Args:
            x: World X position in meters.
            y: World Y position in meters.

        Returns:
            AdaptiveCellInfo with resolution, tier, and elevation data.
        """
        if not self.in_bounds(x, y):
            return AdaptiveCellInfo(query_x=x, query_y=y, is_represented=False)

        base_row, base_col = self._base_grid_index(x, y)
        rows, cols = self.base_shape

        # Safety clamp
        base_row = max(0, min(base_row, rows - 1))
        base_col = max(0, min(base_col, cols - 1))

        tier = self.index_tier[base_row, base_col]
        if tier == "":
            return AdaptiveCellInfo(query_x=x, query_y=y, is_represented=False)

        tier_grid = self.tier_maps[tier]

        # Convert query world coordinates directly to fine-grid indices
        fine_row, fine_col = tier_grid.world_to_grid(x, y)
        fine_row = int(fine_row)
        fine_col = int(fine_col)

        # Validate fine-grid indices
        if not (0 <= fine_row < tier_grid.rows and 0 <= fine_col < tier_grid.cols):
            return AdaptiveCellInfo(query_x=x, query_y=y, is_represented=False)

        count = int(tier_grid.point_count[fine_row, fine_col])
        if count == 0:
            return AdaptiveCellInfo(query_x=x, query_y=y, is_represented=False)

        return AdaptiveCellInfo(
            query_x=x,
            query_y=y,
            is_represented=True,
            tier=str(tier),
            resolution=float(self.tier_resolutions[str(tier)]),
            min_z=float(tier_grid.min_z[fine_row, fine_col]),
            max_z=float(tier_grid.max_z[fine_row, fine_col]),
            mean_z=float(tier_grid.mean_z[fine_row, fine_col]),
            point_count=count,
            mean_intensity=(
                float(tier_grid.mean_intensity[fine_row, fine_col])
                if not np.isnan(tier_grid.mean_intensity[fine_row, fine_col])
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Summary / statistics
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns a structured summary of the adaptive map contents.

        Includes per-tier cell counts, percentages, a uniform ultra-fine
        reference comparison, and estimated memory savings.
        """
        counts = self.tier_cell_counts
        pcts = self.tier_percentages
        total_repr = self.num_represented_cells

        # Hypothetical uniform ultra-fine map cell count
        fine_res = min(self.tier_resolutions.values()) if self.tier_resolutions else self.base_resolution
        span_x = self.max_x - self.min_x
        span_y = self.max_y - self.min_y
        uniform_fine_cells = int(
            np.ceil(span_x / fine_res) * np.ceil(span_y / fine_res)
        )

        # Actual unique tier-grid cells occupied in the adaptive map
        actual_tier_cells = sum(
            grid.num_occupied_cells for grid in self.tier_maps.values()
        )

        return {
            "base_resolution_m": self.base_resolution,
            "base_shape": self.base_shape,
            "base_bounds": {
                "min_x": self.min_x, "max_x": self.max_x,
                "min_y": self.min_y, "max_y": self.max_y,
            },
            "available_tiers": self.available_tiers,
            "tier_resolutions": dict(self.tier_resolutions),
            "represented_base_cells": total_repr,
            "tier_counts": counts,
            "tier_percentages": {t: round(p, 2) for t, p in pcts.items()},
            "actual_tier_occupied_cells": actual_tier_cells,
            "uniform_fine_total_cells": uniform_fine_cells,
            "uniform_fine_resolution_m": fine_res,
        }
