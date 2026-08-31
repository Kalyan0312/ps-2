"""
Unit tests for Terrain Complexity and Roughness Analysis (Phase 4).
"""

import numpy as np
import pytest
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarLoader
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.mapping.grid_map import GridMap25D
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.terrain.config import TerrainConfig
from backend.terrain.result import TerrainAnalysisResult
from backend.terrain.analyzer import TerrainAnalyzer


def create_mock_grid(mean_z_matrix: np.ndarray, resolution: float = 1.0) -> GridMap25D:
    """Helper to build a GridMap25D directly from a 2D matrix."""
    rows, cols = mean_z_matrix.shape
    point_count = np.where(~np.isnan(mean_z_matrix), 1, 0).astype(np.int32)
    min_z = mean_z_matrix.copy().astype(np.float32)
    max_z = mean_z_matrix.copy().astype(np.float32)
    mean_z = mean_z_matrix.copy().astype(np.float32)
    mean_intensity = np.where(~np.isnan(mean_z_matrix), 0.5, np.nan).astype(np.float32)

    return GridMap25D(
        min_x=0.0,
        max_x=float(cols * resolution),
        min_y=0.0,
        max_y=float(rows * resolution),
        resolution=resolution,
        point_count=point_count,
        min_z=min_z,
        max_z=max_z,
        mean_z=mean_z,
        mean_intensity=mean_intensity,
    )


class TestTerrainAnalyzer:
    """Test suite for terrain feature calculations."""

    def test_flat_terrain_low_roughness(self):
        # 5x5 flat ground with z = 1.0
        z_grid = np.full((5, 5), 1.0, dtype=np.float32)
        grid = create_mock_grid(z_grid, resolution=1.0)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3))
        result = analyzer.analyze(grid)

        # Center cell (2, 2)
        center_cell = result.get_cell(2, 2)
        assert center_cell["is_analyzed"] is True
        assert center_cell["roughness"] == pytest.approx(0.0, abs=1e-5)
        assert center_cell["elevation_range"] == pytest.approx(0.0, abs=1e-5)
        assert center_cell["slope"] == pytest.approx(0.0, abs=1e-5)

    def test_uneven_terrain_higher_roughness(self):
        # 5x5 flat ground
        z_flat = np.full((5, 5), 0.0, dtype=np.float32)
        grid_flat = create_mock_grid(z_flat, resolution=1.0)

        # 5x5 uneven terrain with a step obstacle
        z_uneven = np.full((5, 5), 0.0, dtype=np.float32)
        z_uneven[2:, :] = 2.5 # 2.5m elevation step
        grid_uneven = create_mock_grid(z_uneven, resolution=1.0)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3))
        res_flat = analyzer.analyze(grid_flat)
        res_uneven = analyzer.analyze(grid_uneven)

        assert res_uneven.roughness[2, 2] > res_flat.roughness[2, 2]
        assert res_uneven.elevation_range[2, 2] == pytest.approx(2.5)
        assert res_uneven.slope[2, 2] > 0.0

    def test_known_slope_gradient_calculation(self):
        # Ramp z = 2.0 * x (gradient along X is 2.0 m/m)
        # 5x5 grid with resolution 1.0m
        rows, cols = 5, 5
        x_indices = np.arange(cols)
        z_ramp = np.tile(2.0 * x_indices, (rows, 1)).astype(np.float32)
        grid = create_mock_grid(z_ramp, resolution=1.0)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3))
        result = analyzer.analyze(grid)

        # Center cell should have slope ~ 2.0 m/m
        assert result.slope[2, 2] == pytest.approx(2.0, rel=1e-3)

    def test_empty_cell_nan_safety(self):
        # 3x3 grid with empty (NaN) cells around a valid cell
        z_sparse = np.full((3, 3), np.nan, dtype=np.float32)
        z_sparse[1, 1] = 2.0
        z_sparse[1, 2] = 2.0
        grid = create_mock_grid(z_sparse, resolution=1.0)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3, min_valid_neighbors=1))
        result = analyzer.analyze(grid)

        # Occupied cell (1, 1) should be analyzed cleanly without NaNs
        assert bool(result.analyzed_mask[1, 1]) is True
        assert not np.isnan(result.roughness[1, 1])
        assert not np.isnan(result.elevation_range[1, 1])
        assert not np.isnan(result.slope[1, 1])

        # Empty cell (0, 0) should remain NaN and not analyzed
        assert bool(result.analyzed_mask[0, 0]) is False
        assert np.isnan(result.roughness[0, 0])

    def test_boundary_handling(self):
        # Test corner cell (0, 0)
        z_grid = np.full((4, 4), 1.0, dtype=np.float32)
        grid = create_mock_grid(z_grid, resolution=1.0)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3))
        result = analyzer.analyze(grid)

        corner_cell = result.get_cell(0, 0)
        assert corner_cell["is_analyzed"] is True
        assert corner_cell["roughness"] == pytest.approx(0.0, abs=1e-5)

    def test_config_validation(self):
        with pytest.raises(ValueError):
            TerrainConfig(window_size=2) # Even window size invalid
        with pytest.raises(ValueError):
            TerrainConfig(window_size=1) # Window size < 3 invalid
        with pytest.raises(ValueError):
            TerrainConfig(min_valid_neighbors=0)

    def test_yaml_config_loading(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        cfg = TerrainConfig.from_yaml(config_path)

        assert cfg.enabled is True
        assert cfg.window_size == 3
        assert cfg.calculate_roughness is True
        assert cfg.calculate_slope is True

    def test_pipeline_integration(self):
        loader = SyntheticLidarLoader(num_frames=1, seed=42)
        raw_frame = loader[0]

        prep_pipeline = PreprocessingPipeline()
        clean_frame = prep_pipeline.process(raw_frame)

        map_builder = Uniform25DMapBuilder(MappingConfig(resolution=0.20, min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0))
        grid = map_builder.build_map(clean_frame)

        analyzer = TerrainAnalyzer(TerrainConfig(window_size=3))
        result = analyzer.analyze(grid)

        assert result.num_analyzed_cells == grid.num_occupied_cells
        stats = result.get_stats()
        assert stats["roughness"]["max"] >= stats["roughness"]["min"]
        assert stats["slope"]["max"] >= stats["slope"]["min"]
