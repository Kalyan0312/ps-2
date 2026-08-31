"""
Unit tests for Adaptive Resolution Decision Engine (Phase 5).
"""

import numpy as np
import pytest
from pathlib import Path
from typing import Tuple

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarLoader
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.mapping.grid_map import GridMap25D
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.terrain.config import TerrainConfig
from backend.terrain.result import TerrainAnalysisResult
from backend.terrain.analyzer import TerrainAnalyzer
from backend.priority.config import PriorityConfig, DEFAULT_RESOLUTION_LEVELS
from backend.priority.result import ResolutionDecisionResult
from backend.priority.selector import AdaptiveResolutionSelector


def build_mock_terrain_scenario(roughness_val: float, slope_val: float, elev_range_val: float) -> Tuple[GridMap25D, TerrainAnalysisResult]:
    """Helper to mock a 3x3 grid with specific terrain metrics at center cell (1, 1)."""
    rows, cols = 3, 3
    point_count = np.ones((rows, cols), dtype=np.int32)
    mean_z = np.full((rows, cols), 1.0, dtype=np.float32)

    grid = GridMap25D(
        min_x=0.0, max_x=3.0, min_y=0.0, max_y=3.0, resolution=1.0,
        point_count=point_count, min_z=mean_z, max_z=mean_z, mean_z=mean_z,
        mean_intensity=mean_z,
    )

    r_layer = np.full((rows, cols), roughness_val, dtype=np.float32)
    s_layer = np.full((rows, cols), slope_val, dtype=np.float32)
    e_layer = np.full((rows, cols), elev_range_val, dtype=np.float32)
    analyzed_mask = np.ones((rows, cols), dtype=bool)

    terrain_res = TerrainAnalysisResult(
        grid_map=grid,
        roughness=r_layer,
        elevation_range=e_layer,
        slope=s_layer,
        analyzed_mask=analyzed_mask,
    )
    return grid, terrain_res


class TestAdaptiveResolutionSelector:
    """Test suite for adaptive resolution assignment logic."""

    def test_flat_terrain_selects_coarse(self):
        # Low complexity metrics: roughness=0.005, slope=0.05, elev_range=0.02
        grid, terrain_res = build_mock_terrain_scenario(0.005, 0.05, 0.02)
        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_res)

        cell = decision.get_cell(1, 1)
        assert cell["is_decided"] is True
        assert cell["assigned_level"] == "coarse"
        assert cell["assigned_resolution"] == pytest.approx(0.50)

    def test_moderate_terrain_selects_medium(self):
        # Moderate complexity metrics exceeding medium thresholds
        grid, terrain_res = build_mock_terrain_scenario(0.03, 0.20, 0.12)
        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_res)

        cell = decision.get_cell(1, 1)
        assert cell["assigned_level"] == "medium"
        assert cell["assigned_resolution"] == pytest.approx(0.25)

    def test_complex_terrain_selects_fine(self):
        # High complexity metrics exceeding fine thresholds
        grid, terrain_res = build_mock_terrain_scenario(0.12, 0.60, 0.40)
        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_res)

        cell = decision.get_cell(1, 1)
        assert cell["assigned_level"] == "fine"
        assert cell["assigned_resolution"] == pytest.approx(0.10)

    def test_very_complex_terrain_selects_ultra_fine(self):
        # Very high complexity metrics exceeding ultra-fine thresholds
        grid, terrain_res = build_mock_terrain_scenario(0.35, 2.0, 1.2)
        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_res)

        cell = decision.get_cell(1, 1)
        assert cell["assigned_level"] == "ultra_fine"
        assert cell["assigned_resolution"] == pytest.approx(0.05)

    def test_empty_cell_nan_safety(self):
        # 3x3 grid with empty cells
        rows, cols = 3, 3
        point_count = np.zeros((rows, cols), dtype=np.int32)
        point_count[1, 1] = 5 # only center is occupied

        mean_z = np.full((rows, cols), np.nan, dtype=np.float32)
        mean_z[1, 1] = 1.0

        grid = GridMap25D(
            min_x=0.0, max_x=3.0, min_y=0.0, max_y=3.0, resolution=1.0,
            point_count=point_count, min_z=mean_z, max_z=mean_z, mean_z=mean_z,
            mean_intensity=mean_z,
        )

        r_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        r_layer[1, 1] = 0.01
        s_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        s_layer[1, 1] = 0.05
        e_layer = np.full((rows, cols), np.nan, dtype=np.float32)
        e_layer[1, 1] = 0.02
        analyzed_mask = np.zeros((rows, cols), dtype=bool)
        analyzed_mask[1, 1] = True

        terrain_res = TerrainAnalysisResult(
            grid_map=grid, roughness=r_layer, elevation_range=e_layer,
            slope=s_layer, analyzed_mask=analyzed_mask,
        )

        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_res)

        # Center cell should be coarse
        assert decision.assigned_level[1, 1] == "coarse"
        assert bool(decision.valid_mask[1, 1]) is True

        # Empty cell (0, 0) should be NaN and not decided
        assert bool(decision.valid_mask[0, 0]) is False
        assert np.isnan(decision.assigned_resolution[0, 0])
        assert decision.assigned_level[0, 0] == ""

    def test_yaml_config_loading(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        cfg = PriorityConfig.from_yaml(config_path)

        assert cfg.enabled is True
        assert "ultra_fine" in cfg.thresholds
        assert cfg.thresholds["ultra_fine"]["roughness"] == pytest.approx(0.25)
        assert cfg.resolution_levels["coarse"] == pytest.approx(0.50)

    def test_full_pipeline_integration(self):
        loader = SyntheticLidarLoader(num_frames=1, seed=42)
        raw_frame = loader[0]

        prep_pipeline = PreprocessingPipeline()
        clean_frame = prep_pipeline.process(raw_frame)

        map_builder = Uniform25DMapBuilder(MappingConfig(resolution=0.20, min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0))
        grid = map_builder.build_map(clean_frame)

        analyzer = TerrainAnalyzer()
        terrain_result = analyzer.analyze(grid)

        selector = AdaptiveResolutionSelector()
        decision = selector.select_resolution(grid, terrain_result)

        assert decision.num_decided_cells == grid.num_occupied_cells
        counts = decision.counts
        assert sum(counts.values()) == decision.num_decided_cells

        # Both flat ground (coarse) and obstacle (fine/ultra-fine) should be detected in synthetic frame
        assert counts["coarse"] > 0
        assert (counts["fine"] + counts["ultra_fine"]) > 0
