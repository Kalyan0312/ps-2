"""
Deterministic Resolution Cost Model.

Estimates relative spatial cell allocation, memory consumption, and compute cost
for different 2.5D elevation grid resolution tiers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


# Standard downgrade progression hierarchy
DOWNGRADE_ORDER: Dict[str, Optional[str]] = {
    "ultra_fine": "fine",
    "fine": "medium",
    "medium": "coarse",
    "coarse": None,
}

# GridMap25D physical array storage layout:
# 5 layers (1x int32 point_count, 4x float32 [min_z, max_z, mean_z, mean_intensity]) * 4 bytes = 20 bytes/cell
ACTUAL_BYTES_PER_GRID_CELL = 20
BYTES_PER_GRID_CELL = ACTUAL_BYTES_PER_GRID_CELL


class ResolutionCostModel:
    """
    Cost model providing deterministic resource and array memory estimates across resolution levels.

    Model Definitions:
      1. Spatial scaling: cell density scales as (base_resolution / tier_resolution)²
      2. Memory footprint: estimated physical array bytes based on 20 bytes/cell
         (1x int32 layer + 4x float32 layers).
      3. Compute cost: abstract compute workload units scaling with fine cell density.
    """

    def __init__(self, bytes_per_cell: int = BYTES_PER_GRID_CELL):
        self.bytes_per_cell = bytes_per_cell

    def get_cell_multiplier(self, tier_resolution: float, base_resolution: float = 0.20) -> float:
        """
        Calculates the ratio of fine-grid cells per base-grid cell.

        Example:
          base_res = 0.20 m, tier_res = 0.05 m -> (0.20 / 0.05)² = 16.0 cells / base cell.
          base_res = 0.20 m, tier_res = 0.50 m -> (0.20 / 0.50)² = 0.16 cells / base cell.
        """
        if tier_resolution <= 0 or base_resolution <= 0:
            raise ValueError(f"Resolutions must be positive: tier={tier_resolution}, base={base_resolution}")
        return float((base_resolution / tier_resolution) ** 2)

    def estimate_tier_cells(
        self,
        num_base_cells: int,
        tier_resolution: float,
        base_resolution: float = 0.20,
    ) -> int:
        """
        Estimates the number of allocated fine-grid cells for `num_base_cells` assigned to this tier.
        """
        if num_base_cells <= 0:
            return 0
        multiplier = self.get_cell_multiplier(tier_resolution, base_resolution)
        return int(np.ceil(num_base_cells * multiplier))

    def estimate_tier_memory_bytes(
        self,
        num_base_cells: int,
        tier_resolution: float,
        base_resolution: float = 0.20,
    ) -> int:
        """
        Estimates the array memory in bytes for `num_base_cells` assigned to this tier.
        """
        cells = self.estimate_tier_cells(num_base_cells, tier_resolution, base_resolution)
        return cells * self.bytes_per_cell

    def estimate_tier_compute_cost(
        self,
        num_base_cells: int,
        tier_resolution: float,
        base_resolution: float = 0.20,
    ) -> float:
        """
        Estimates relative compute cost units for spatial binning and statistics calculation.
        """
        if num_base_cells <= 0:
            return 0.0
        # Compute cost scales linearly with fine cell density
        multiplier = self.get_cell_multiplier(tier_resolution, base_resolution)
        return float(num_base_cells * multiplier)

    @staticmethod
    def get_downgrade_target(tier_name: str) -> Optional[str]:
        """Returns the next lower resolution tier in the hierarchy, or None if already coarse."""
        return DOWNGRADE_ORDER.get(tier_name.lower().strip())
