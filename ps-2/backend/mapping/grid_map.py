"""
2.5D Elevation and Grid Map Data Structures.
Represents uniform 2.5D spatial grid maps storing elevation statistics and occupancy per cell.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np


@dataclass
class GridMap25D:
    """
    Uniform 2.5D Elevation Grid Map.
    
    Attributes:
        min_x (float): Minimum X coordinate bound (meters).
        max_x (float): Maximum X coordinate bound (meters).
        min_y (float): Minimum Y coordinate bound (meters).
        max_y (float): Maximum Y coordinate bound (meters).
        resolution (float): Cell dimension in meters.
        point_count (np.ndarray): 2D array of point counts per cell (shape: [rows, cols]).
        min_z (np.ndarray): 2D array of minimum Z per cell (NaN for empty cells).
        max_z (np.ndarray): 2D array of maximum Z per cell (NaN for empty cells).
        mean_z (np.ndarray): 2D array of mean Z per cell (NaN for empty cells).
        mean_intensity (np.ndarray): 2D array of mean intensity (NaN for empty cells).
        metadata (dict): Optional metadata (frame_id, timestamp, etc.).
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution: float
    point_count: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    mean_z: np.ndarray
    mean_intensity: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {self.resolution}")
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError(
                f"Invalid bounds: X [{self.min_x}, {self.max_x}], Y [{self.min_y}, {self.max_y}]"
            )

        expected_rows = int(np.ceil((self.max_y - self.min_y) / self.resolution))
        expected_cols = int(np.ceil((self.max_x - self.min_x) / self.resolution))

        for name, layer in [
            ("point_count", self.point_count),
            ("min_z", self.min_z),
            ("max_z", self.max_z),
            ("mean_z", self.mean_z),
            ("mean_intensity", self.mean_intensity),
        ]:
            if layer.shape != (expected_rows, expected_cols):
                raise ValueError(
                    f"Layer '{name}' shape {layer.shape} does not match expected shape ({expected_rows}, {expected_cols})"
                )

    @property
    def rows(self) -> int:
        """Number of grid rows (along Y)."""
        return self.point_count.shape[0]

    @property
    def cols(self) -> int:
        """Number of grid columns (along X)."""
        return self.point_count.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Dimensions of the grid (rows, cols)."""
        return self.rows, self.cols

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.rows * self.cols

    @property
    def occupied_mask(self) -> np.ndarray:
        """Boolean 2D mask indicating occupied cells."""
        return self.point_count > 0

    @property
    def num_occupied_cells(self) -> int:
        """Count of occupied cells containing >= 1 point."""
        return int(np.count_nonzero(self.occupied_mask))

    @property
    def num_empty_cells(self) -> int:
        """Count of empty cells containing 0 points."""
        return self.total_cells - self.num_occupied_cells

    @property
    def occupancy_ratio(self) -> float:
        """Fraction of total cells that are occupied [0.0, 1.0]."""
        return self.num_occupied_cells / self.total_cells if self.total_cells > 0 else 0.0

    @property
    def occupancy_percentage(self) -> float:
        """Percentage of total cells that are occupied [0.0%, 100.0%]."""
        return self.occupancy_ratio * 100.0

    @property
    def elevation_difference(self) -> np.ndarray:
        """2D array of (max_z - min_z) per cell (NaN for empty cells)."""
        return self.max_z - self.min_z

    def world_to_grid(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Converts world XY coordinates (meters) to grid (row, col) indices.
        Row corresponds to Y index, Col corresponds to X index.
        Clamps exact upper boundary coordinates into the last valid grid cell.
        """
        col = np.floor((x - self.min_x) / self.resolution).astype(int)
        row = np.floor((y - self.min_y) / self.resolution).astype(int)
        if isinstance(col, np.ndarray):
            col = np.clip(col, 0, self.cols - 1)
        else:
            col = max(0, min(int(col), self.cols - 1))
        if isinstance(row, np.ndarray):
            row = np.clip(row, 0, self.rows - 1)
        else:
            row = max(0, min(int(row), self.rows - 1))
        return row, col

    def grid_to_world(
        self,
        row: Union[int, np.ndarray],
        col: Union[int, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Converts grid (row, col) indices to world XY coordinates (cell center in meters).
        """
        x = self.min_x + (col + 0.5) * self.resolution
        y = self.min_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> Union[bool, np.ndarray]:
        """Checks if world XY coordinates fall within map bounds (inclusive)."""
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return (x >= self.min_x) & (x <= self.max_x) & (y >= self.min_y) & (y <= self.max_y)
        return bool(self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y)

    def is_occupied(self, row: int, col: int) -> bool:
        """Checks if a specific grid cell contains points."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return bool(self.point_count[row, col] > 0)

    def get_cell(self, row: int, col: int) -> Dict[str, Any]:
        """Returns elevation and stats dictionary for a specific cell."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(f"Cell ({row}, {col}) out of grid bounds {self.shape}")
        
        count = int(self.point_count[row, col])
        is_occ = count > 0
        cx, cy = self.grid_to_world(row, col)

        return {
            "row": row,
            "col": col,
            "center_x": float(cx),
            "center_y": float(cy),
            "is_occupied": is_occ,
            "point_count": count,
            "min_z": float(self.min_z[row, col]) if is_occ else None,
            "max_z": float(self.max_z[row, col]) if is_occ else None,
            "mean_z": float(self.mean_z[row, col]) if is_occ else None,
            "mean_intensity": float(self.mean_intensity[row, col]) if is_occ else None,
            "elevation_diff": float(self.elevation_difference[row, col]) if is_occ else None,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level summary of map metrics."""
        return {
            "resolution": self.resolution,
            "bounds": {
                "min_x": self.min_x,
                "max_x": self.max_x,
                "min_y": self.min_y,
                "max_y": self.max_y,
            },
            "dimensions": {"rows": self.rows, "cols": self.cols},
            "total_cells": self.total_cells,
            "occupied_cells": self.num_occupied_cells,
            "occupancy_percentage": self.occupancy_percentage,
            "min_elevation": float(np.nanmin(self.min_z)) if self.num_occupied_cells > 0 else None,
            "max_elevation": float(np.nanmax(self.max_z)) if self.num_occupied_cells > 0 else None,
        }
