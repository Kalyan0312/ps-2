"""
Resolution Decision Result Data Structure.
Encapsulates spatial resolution level assignments and allocation distributions across the 2.5D grid.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.terrain.result import TerrainAnalysisResult


@dataclass
class ResolutionDecisionResult:
    """
    Stores spatial resolution assignments decided by terrain complexity metrics.
    
    Attributes:
        grid_map (GridMap25D): Source 2.5D elevation grid map.
        terrain_result (TerrainAnalysisResult): Computed terrain complexity features.
        assigned_resolution (np.ndarray): 2D float32 array of cell resolution in meters (NaN for empty/unanalyzed).
        assigned_level (np.ndarray): 2D string array of level names ('coarse', 'medium', 'fine', 'ultra_fine', or '').
        valid_mask (np.ndarray): Boolean mask indicating analyzed cells with valid resolution assignments.
        resolution_levels (Dict[str, float]): Registered resolution level values in meters.
        metadata (dict): Decision metadata and thresholds used.
    """
    grid_map: GridMap25D
    terrain_result: TerrainAnalysisResult
    assigned_resolution: np.ndarray
    assigned_level: np.ndarray
    valid_mask: np.ndarray
    resolution_levels: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        expected_shape = self.grid_map.shape
        for name, layer in [
            ("assigned_resolution", self.assigned_resolution),
            ("assigned_level", self.assigned_level),
            ("valid_mask", self.valid_mask),
        ]:
            if layer.shape != expected_shape:
                raise ValueError(
                    f"Layer '{name}' shape {layer.shape} does not match grid shape {expected_shape}"
                )

    @property
    def rows(self) -> int:
        return self.grid_map.rows

    @property
    def cols(self) -> int:
        return self.grid_map.cols

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid_map.shape

    @property
    def num_decided_cells(self) -> int:
        """Total number of occupied and analyzed cells with resolution decisions."""
        return int(np.count_nonzero(self.valid_mask))

    @property
    def counts(self) -> Dict[str, int]:
        """Counts of assigned cells for each resolution level tier."""
        tier_counts = {tier: 0 for tier in self.resolution_levels.keys()}
        if self.num_decided_cells == 0:
            return tier_counts

        valid_levels = self.assigned_level[self.valid_mask]
        for tier in tier_counts.keys():
            tier_counts[tier] = int(np.count_nonzero(valid_levels == tier))
        return tier_counts

    @property
    def percentages(self) -> Dict[str, float]:
        """Percentage distribution of assigned resolution levels over decided cells."""
        total = self.num_decided_cells
        if total == 0:
            return {tier: 0.0 for tier in self.resolution_levels.keys()}
        return {
            tier: (count / total) * 100.0 for tier, count in self.counts.items()
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured summary of resolution assignment counts and percentages."""
        return {
            "total_occupied_cells": self.grid_map.num_occupied_cells,
            "total_decided_cells": self.num_decided_cells,
            "counts": self.counts,
            "percentages": self.percentages,
            "resolution_levels": self.resolution_levels,
        }

    def get_cell(self, row: int, col: int) -> Dict[str, Any]:
        """Returns full diagnostic and resolution decision for a single grid cell."""
        cell_info = self.terrain_result.get_cell(row, col)
        is_decided = bool(self.valid_mask[row, col])
        cell_info.update({
            "is_decided": is_decided,
            "assigned_resolution": float(self.assigned_resolution[row, col]) if is_decided else None,
            "assigned_level": str(self.assigned_level[row, col]) if is_decided else None,
        })
        return cell_info
