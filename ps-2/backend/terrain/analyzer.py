"""
Terrain Analyzer Implementation.
Computes local roughness (std dev), elevation range, and gradient slopes from a 2.5D elevation map.
"""

from typing import Optional
from pathlib import Path
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.terrain.config import TerrainConfig
from backend.terrain.result import TerrainAnalysisResult


class TerrainAnalyzer:
    """
    Analyzes terrain complexity metrics across a GridMap25D.
    """

    def __init__(self, config: Optional[TerrainConfig] = None):
        self.config = config or TerrainConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "TerrainAnalyzer":
        """Instantiates analyzer directly from YAML configuration file."""
        cfg = TerrainConfig.from_yaml(config_path)
        return cls(config=cfg)

    def analyze(self, grid_map: GridMap25D) -> TerrainAnalysisResult:
        """
        Extracts spatial terrain features from the input 2.5D grid map.
        
        Args:
            grid_map (GridMap25D): Source elevation map.
            
        Returns:
            TerrainAnalysisResult: Computed terrain feature layers.
        """
        rows, cols = grid_map.shape
        res = grid_map.resolution
        mean_z = grid_map.mean_z
        occupied = grid_map.occupied_mask
        w = self.config.window_size
        half_w = w // 2
        min_neighbors = self.config.min_valid_neighbors

        # Initialize output feature layers (NaN for unanalyzed / empty cells)
        roughness_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        elevation_range_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        slope_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        analyzed_mask = np.zeros((rows, cols), dtype=bool)

        if not self.config.enabled or grid_map.num_occupied_cells == 0:
            return TerrainAnalysisResult(
                grid_map=grid_map,
                roughness=roughness_layer,
                elevation_range=elevation_range_layer,
                slope=slope_layer,
                analyzed_mask=analyzed_mask,
                metadata={"status": "disabled_or_empty"},
            )

        # Vectorized sliding window extraction
        padded = np.pad(mean_z, half_w, mode="constant", constant_values=np.nan)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (w, w))

        valid_mask = ~np.isnan(windows)
        valid_counts = np.sum(valid_mask, axis=(-2, -1))
        valid_cells = occupied & (valid_counts >= min_neighbors)

        if np.any(valid_cells):
            occ_windows = windows[valid_cells]

            # 1. Local Elevation Roughness (Standard Deviation)
            if self.config.calculate_roughness:
                roughness_layer[valid_cells] = np.nanstd(occ_windows, axis=(1, 2)).astype(np.float32)

            # 2. Local Elevation Range (Max - Min)
            if self.config.calculate_elevation_range:
                min_w = np.nanmin(occ_windows, axis=(1, 2))
                max_w = np.nanmax(occ_windows, axis=(1, 2))
                elevation_range_layer[valid_cells] = (max_w - min_w).astype(np.float32)

            # 3. Local Slope (Gradient Magnitude)
            if self.config.calculate_slope:
                left = np.pad(mean_z[:, :-1], ((0, 0), (1, 0)), constant_values=np.nan)
                right = np.pad(mean_z[:, 1:], ((0, 0), (0, 1)), constant_values=np.nan)
                up = np.pad(mean_z[:-1, :], ((1, 0), (0, 0)), constant_values=np.nan)
                down = np.pad(mean_z[1:, :], ((0, 1), (0, 0)), constant_values=np.nan)

                has_l = ~np.isnan(left)
                has_r = ~np.isnan(right)
                has_u = ~np.isnan(up)
                has_d = ~np.isnan(down)

                dz_dx = np.zeros((rows, cols), dtype=np.float32)
                both_lr = has_l & has_r
                only_r = (~has_l) & has_r
                only_l = has_l & (~has_r)

                dz_dx[both_lr] = (right[both_lr] - left[both_lr]) / (2.0 * res)
                dz_dx[only_r] = (right[only_r] - mean_z[only_r]) / res
                dz_dx[only_l] = (mean_z[only_l] - left[only_l]) / res

                dz_dy = np.zeros((rows, cols), dtype=np.float32)
                both_ud = has_u & has_d
                only_d = (~has_u) & has_d
                only_u = has_u & (~has_d)

                dz_dy[both_ud] = (down[both_ud] - up[both_ud]) / (2.0 * res)
                dz_dy[only_d] = (down[only_d] - mean_z[only_d]) / res
                dz_dy[only_u] = (mean_z[only_u] - up[only_u]) / res

                slope_full = np.sqrt(dz_dx ** 2 + dz_dy ** 2).astype(np.float32)
                slope_layer[valid_cells] = slope_full[valid_cells]

            analyzed_mask[valid_cells] = True

        metadata = {
            "window_size": self.config.window_size,
            "min_valid_neighbors": min_neighbors,
            "num_occupied": grid_map.num_occupied_cells,
            "num_analyzed": int(np.count_nonzero(analyzed_mask)),
        }

        return TerrainAnalysisResult(
            grid_map=grid_map,
            roughness=roughness_layer,
            elevation_range=elevation_range_layer,
            slope=slope_layer,
            analyzed_mask=analyzed_mask,
            metadata=metadata,
        )
