"""
Terrain Analysis Result Data Structure.
Encapsulates computed terrain features (roughness, elevation range, slope) across a 2.5D grid map.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np

from backend.mapping.grid_map import GridMap25D


@dataclass
class TerrainAnalysisResult:
    """
    Stores spatial terrain complexity feature layers computed from a GridMap25D.
    
    Attributes:
        grid_map (GridMap25D): The source elevation map.
        roughness (np.ndarray): Local elevation standard deviation (NaN for unanalyzed cells).
        elevation_range (np.ndarray): Local elevation difference max - min (NaN for unanalyzed cells).
        slope (np.ndarray): Local gradient magnitude dz/dr (NaN for unanalyzed cells).
        analyzed_mask (np.ndarray): Boolean mask indicating cells successfully analyzed.
        metadata (dict): Processing metadata (window_size, runtime, etc.).
    """
    grid_map: GridMap25D
    roughness: np.ndarray
    elevation_range: np.ndarray
    slope: np.ndarray
    analyzed_mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        expected_shape = self.grid_map.shape
        for name, layer in [
            ("roughness", self.roughness),
            ("elevation_range", self.elevation_range),
            ("slope", self.slope),
            ("analyzed_mask", self.analyzed_mask),
        ]:
            if layer.shape != expected_shape:
                raise ValueError(
                    f"Layer '{name}' shape {layer.shape} does not match grid map shape {expected_shape}"
                )

    @property
    def rows(self) -> int:
        """Number of rows in the grid."""
        return self.grid_map.rows

    @property
    def cols(self) -> int:
        """Number of columns in the grid."""
        return self.grid_map.cols

    @property
    def shape(self) -> Tuple[int, int]:
        """Dimensions of the grid (rows, cols)."""
        return self.grid_map.shape

    @property
    def num_analyzed_cells(self) -> int:
        """Total count of cells with valid terrain feature calculations."""
        return int(np.count_nonzero(self.analyzed_mask))

    def get_stats(self) -> Dict[str, Any]:
        """
        Computes aggregate statistics (min, max, mean) for each terrain feature layer
        over analyzed cells.
        """
        if self.num_analyzed_cells == 0:
            return {
                "num_analyzed_cells": 0,
                "roughness": {"min": 0.0, "max": 0.0, "mean": 0.0},
                "elevation_range": {"min": 0.0, "max": 0.0, "mean": 0.0},
                "slope": {"min": 0.0, "max": 0.0, "mean": 0.0},
            }

        r_valid = self.roughness[self.analyzed_mask]
        e_valid = self.elevation_range[self.analyzed_mask]
        s_valid = self.slope[self.analyzed_mask]

        return {
            "num_analyzed_cells": self.num_analyzed_cells,
            "roughness": {
                "min": float(np.nanmin(r_valid)) if len(r_valid) > 0 else 0.0,
                "max": float(np.nanmax(r_valid)) if len(r_valid) > 0 else 0.0,
                "mean": float(np.nanmean(r_valid)) if len(r_valid) > 0 else 0.0,
            },
            "elevation_range": {
                "min": float(np.nanmin(e_valid)) if len(e_valid) > 0 else 0.0,
                "max": float(np.nanmax(e_valid)) if len(e_valid) > 0 else 0.0,
                "mean": float(np.nanmean(e_valid)) if len(e_valid) > 0 else 0.0,
            },
            "slope": {
                "min": float(np.nanmin(s_valid)) if len(s_valid) > 0 else 0.0,
                "max": float(np.nanmax(s_valid)) if len(s_valid) > 0 else 0.0,
                "mean": float(np.nanmean(s_valid)) if len(s_valid) > 0 else 0.0,
            },
        }

    def get_cell(self, row: int, col: int) -> Dict[str, Any]:
        """Returns terrain features for a specific cell index."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"Cell ({row}, {col}) out of bounds {self.shape}")

        is_valid = bool(self.analyzed_mask[row, col])
        cell_info = self.grid_map.get_cell(row, col)
        cell_info.update({
            "is_analyzed": is_valid,
            "roughness": float(self.roughness[row, col]) if is_valid else None,
            "elevation_range": float(self.elevation_range[row, col]) if is_valid else None,
            "slope": float(self.slope[row, col]) if is_valid else None,
        })
        return cell_info
