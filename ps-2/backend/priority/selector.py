"""
Adaptive Resolution Decision Selector.
Assigns spatial resolution levels based on terrain roughness, slope, and elevation range.
"""

from typing import Optional
from pathlib import Path
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.terrain.result import TerrainAnalysisResult
from backend.priority.config import PriorityConfig
from backend.priority.result import ResolutionDecisionResult


class AdaptiveResolutionSelector:
    """
    Evaluates terrain complexity metrics and determines the appropriate
    mapping resolution for each spatial grid cell.
    
    Decision Architecture:
    ----------------------
    For each analyzed grid cell, the selector evaluates 3 terrain complexity metrics:
      1. Roughness (Local elevation standard deviation)
      2. Slope (Local elevation gradient magnitude dz/dr)
      3. Elevation Range (Local max - min elevation difference)
      
    Rule Hierarchy:
      - Ultra-Fine (e.g. 0.05m): Triggered if roughness >= T_uf, slope >= T_uf, or elevation_range >= T_uf.
      - Fine (e.g. 0.10m): Triggered if roughness >= T_fine, slope >= T_fine, or elevation_range >= T_fine.
      - Medium (e.g. 0.25m): Triggered if roughness >= T_med, slope >= T_med, or elevation_range >= T_med.
      - Coarse (e.g. 0.50m): Baseline assigned to all remaining low-complexity / flat terrain cells.
    """

    def __init__(self, config: Optional[PriorityConfig] = None):
        self.config = config or PriorityConfig()

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "AdaptiveResolutionSelector":
        """Instantiates selector directly from YAML configuration file."""
        cfg = PriorityConfig.from_yaml(config_path)
        return cls(config=cfg)

    def select_resolution(
        self,
        grid_map: GridMap25D,
        terrain_result: TerrainAnalysisResult,
    ) -> ResolutionDecisionResult:
        """
        Executes resolution assignment across the grid map based on terrain analysis.
        
        Args:
            grid_map (GridMap25D): Base elevation map.
            terrain_result (TerrainAnalysisResult): Extracted terrain features.
            
        Returns:
            ResolutionDecisionResult: Matrix of assigned resolutions and level tiers.
        """
        rows, cols = grid_map.shape
        res_levels = self.config.resolution_levels
        thresholds = self.config.thresholds

        # Initialize decision layers
        assigned_res = np.full((rows, cols), np.nan, dtype=np.float32)
        assigned_level = np.full((rows, cols), "", dtype=object)
        valid_mask = np.zeros((rows, cols), dtype=bool)

        if not self.config.enabled:
            return ResolutionDecisionResult(
                grid_map=grid_map,
                terrain_result=terrain_result,
                assigned_resolution=assigned_res,
                assigned_level=assigned_level,
                valid_mask=valid_mask,
                resolution_levels=res_levels,
                metadata={"status": "disabled"},
            )

        analyzed_mask = terrain_result.analyzed_mask
        roughness = terrain_result.roughness
        slope = terrain_result.slope
        elev_range = terrain_result.elevation_range

        # Thresholds
        t_uf = thresholds.get("ultra_fine", {})
        t_fine = thresholds.get("fine", {})
        t_med = thresholds.get("medium", {})

        # Valid cells: analyzed and containing finite numbers across all features
        valid = analyzed_mask & (~np.isnan(roughness)) & (~np.isnan(slope)) & (~np.isnan(elev_range))

        # 4. Baseline: Coarse (low complexity / flat terrain)
        if "coarse" in res_levels:
            assigned_level[valid] = "coarse"
            assigned_res[valid] = res_levels["coarse"]

        # 3. Medium (moderate complexity / uneven ground)
        if "medium" in res_levels:
            is_medium = valid & (
                (roughness >= t_med.get("roughness", np.inf)) |
                (slope >= t_med.get("slope", np.inf)) |
                (elev_range >= t_med.get("elevation_range", np.inf))
            )
            assigned_level[is_medium] = "medium"
            assigned_res[is_medium] = res_levels["medium"]

        # 2. Fine (high complexity / obstacle boundaries)
        if "fine" in res_levels:
            is_fine = valid & (
                (roughness >= t_fine.get("roughness", np.inf)) |
                (slope >= t_fine.get("slope", np.inf)) |
                (elev_range >= t_fine.get("elevation_range", np.inf))
            )
            assigned_level[is_fine] = "fine"
            assigned_res[is_fine] = res_levels["fine"]

        # 1. Ultra-Fine (very high complexity / steep obstacles)
        if "ultra_fine" in res_levels:
            is_ultra_fine = valid & (
                (roughness >= t_uf.get("roughness", np.inf)) |
                (slope >= t_uf.get("slope", np.inf)) |
                (elev_range >= t_uf.get("elevation_range", np.inf))
            )
            assigned_level[is_ultra_fine] = "ultra_fine"
            assigned_res[is_ultra_fine] = res_levels["ultra_fine"]

        valid_mask = valid

        metadata = {
            "thresholds_used": thresholds,
            "resolution_levels": res_levels,
            "total_analyzed": terrain_result.num_analyzed_cells,
            "total_decided": int(np.count_nonzero(valid_mask)),
        }

        return ResolutionDecisionResult(
            grid_map=grid_map,
            terrain_result=terrain_result,
            assigned_resolution=assigned_res,
            assigned_level=assigned_level,
            valid_mask=valid_mask,
            resolution_levels=res_levels,
            metadata=metadata,
        )
