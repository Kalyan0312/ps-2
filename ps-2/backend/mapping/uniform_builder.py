"""
Uniform 2.5D Elevation Grid Map Builder.
Converts a preprocessed LidarFrame into a uniform GridMap25D.
"""

from typing import Optional
from pathlib import Path
import numpy as np

from backend.data.frame import LidarFrame
from backend.mapping.config import MappingConfig
from backend.mapping.grid_map import GridMap25D


class Uniform25DMapBuilder:
    """
    Constructs a uniform 2.5D elevation grid map from a LiDAR point cloud.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "Uniform25DMapBuilder":
        """Instantiates builder directly from YAML configuration file."""
        cfg = MappingConfig.from_yaml(config_path)
        return cls(config=cfg)

    def build_map(self, frame: LidarFrame) -> GridMap25D:
        """
        Builds a GridMap25D from the given LidarFrame.
        
        Args:
            frame (LidarFrame): Preprocessed point cloud frame.
            
        Returns:
            GridMap25D: Generated 2.5D elevation map.
        """
        res = self.config.resolution
        pts = frame.points

        # Determine spatial boundaries
        if self.config.auto_bounds and len(pts) > 0:
            min_x = float(np.floor(np.min(pts[:, 0]) / res) * res)
            max_x = float(np.floor(np.max(pts[:, 0]) / res + 1.0) * res)
            min_y = float(np.floor(np.min(pts[:, 1]) / res) * res)
            max_y = float(np.floor(np.max(pts[:, 1]) / res + 1.0) * res)
            # Ensure non-zero span
            if max_x <= min_x:
                max_x = min_x + res
            if max_y <= min_y:
                max_y = min_y + res
        else:
            min_x = self.config.min_x
            max_x = self.config.max_x
            min_y = self.config.min_y
            max_y = self.config.max_y

        rows = int(np.ceil((max_y - min_y) / res))
        cols = int(np.ceil((max_x - min_x) / res))
        total_cells = rows * cols

        # Initialize flat grid accumulator layers
        count_flat = np.zeros(total_cells, dtype=np.int32)
        min_z_flat = np.full(total_cells, np.inf, dtype=np.float32)
        max_z_flat = np.full(total_cells, -np.inf, dtype=np.float32)
        sum_z_flat = np.zeros(total_cells, dtype=np.float64)
        sum_intensity_flat = np.zeros(total_cells, dtype=np.float64)

        if len(pts) > 0:
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            z_vals = pts[:, 2].astype(np.float32)
            intensity_vals = pts[:, 3].astype(np.float64)

            # Spatial boundary filter (include points on upper bound by clipping to valid cells)
            valid_mask = (
                (x_vals >= min_x) & (x_vals <= max_x) &
                (y_vals >= min_y) & (y_vals <= max_y)
            )

            if np.any(valid_mask):
                x_valid = x_vals[valid_mask]
                y_valid = y_vals[valid_mask]
                z_valid = z_vals[valid_mask]
                i_valid = intensity_vals[valid_mask]

                # Compute 2D grid indices
                col_indices = np.floor((x_valid - min_x) / res).astype(np.int64)
                row_indices = np.floor((y_valid - min_y) / res).astype(np.int64)

                # Clamp safety to grid limits [0, cols-1], [0, rows-1]
                col_indices = np.clip(col_indices, 0, cols - 1)
                row_indices = np.clip(row_indices, 0, rows - 1)

                # 1D linear cell index
                cell_indices = row_indices * cols + col_indices

                # Vectorized aggregation using ufunc.at
                np.minimum.at(min_z_flat, cell_indices, z_valid)
                np.maximum.at(max_z_flat, cell_indices, z_valid)
                np.add.at(sum_z_flat, cell_indices, z_valid.astype(np.float64))
                np.add.at(sum_intensity_flat, cell_indices, i_valid)
                np.add.at(count_flat, cell_indices, 1)

        # Handle empty cells (set to NaN)
        empty_mask = count_flat == 0
        min_z_flat[empty_mask] = np.nan
        max_z_flat[empty_mask] = np.nan

        mean_z_flat = np.where(count_flat > 0, (sum_z_flat / np.maximum(count_flat, 1)).astype(np.float32), np.nan)
        mean_intensity_flat = np.where(
            count_flat > 0,
            (sum_intensity_flat / np.maximum(count_flat, 1)).astype(np.float32),
            np.nan,
        )

        # Reshape to (rows, cols) 2D matrices
        point_count_2d = count_flat.reshape((rows, cols))
        min_z_2d = min_z_flat.reshape((rows, cols))
        max_z_2d = max_z_flat.reshape((rows, cols))
        mean_z_2d = mean_z_flat.reshape((rows, cols))
        mean_intensity_2d = mean_intensity_flat.reshape((rows, cols))

        metadata = {
            "source_frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "input_points": len(pts),
            "mapped_points": int(np.sum(count_flat)),
        }

        return GridMap25D(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            resolution=res,
            point_count=point_count_2d,
            min_z=min_z_2d,
            max_z=max_z_2d,
            mean_z=mean_z_2d,
            mean_intensity=mean_intensity_2d,
            metadata=metadata,
        )
