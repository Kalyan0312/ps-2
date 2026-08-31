"""
Unit tests for Uniform 2.5D Elevation Grid Mapping (Phase 3A).
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


class TestGridMap25D:
    """Test suite for GridMap25D data structure."""

    def test_grid_initialization_and_properties(self):
        rows, cols = 10, 20
        point_count = np.zeros((rows, cols), dtype=np.int32)
        min_z = np.full((rows, cols), np.nan, dtype=np.float32)
        max_z = np.full((rows, cols), np.nan, dtype=np.float32)
        mean_z = np.full((rows, cols), np.nan, dtype=np.float32)
        mean_intensity = np.full((rows, cols), np.nan, dtype=np.float32)

        # Mark one cell as occupied
        point_count[2, 3] = 5
        min_z[2, 3] = 0.5
        max_z[2, 3] = 1.5
        mean_z[2, 3] = 1.0
        mean_intensity[2, 3] = 0.7

        grid = GridMap25D(
            min_x=-10.0,
            max_x=10.0,
            min_y=-5.0,
            max_y=5.0,
            resolution=1.0,
            point_count=point_count,
            min_z=min_z,
            max_z=max_z,
            mean_z=mean_z,
            mean_intensity=mean_intensity,
        )

        assert grid.shape == (10, 20)
        assert grid.total_cells == 200
        assert grid.num_occupied_cells == 1
        assert grid.num_empty_cells == 199
        assert grid.occupancy_percentage == pytest.approx(0.5)
        assert grid.is_occupied(2, 3) is True
        assert grid.is_occupied(0, 0) is False

    def test_coordinate_transforms(self):
        grid = Uniform25DMapBuilder(
            MappingConfig(resolution=0.5, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        # World to grid
        row, col = grid.world_to_grid(-10.0, -10.0)
        assert row == 0 and col == 0

        row, col = grid.world_to_grid(0.0, 0.0)
        assert row == 20 and col == 20

        # Grid to world (center of cell)
        cx, cy = grid.grid_to_world(0, 0)
        assert cx == pytest.approx(-9.75)
        assert cy == pytest.approx(-9.75)

    def test_in_bounds(self):
        grid = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=-5.0, max_x=5.0, min_y=-5.0, max_y=5.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        assert grid.in_bounds(0.0, 0.0) is True
        assert grid.in_bounds(-5.0, -5.0) is True
        assert grid.in_bounds(10.0, 0.0) is False

    def test_get_cell_info(self):
        pts = np.array([
            [1.2, 2.3, 0.4, 0.2],
            [1.4, 2.1, 1.6, 0.8],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)
        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0)
        )
        grid = builder.build_map(frame)

        # Cell at row=2 (Y in [2, 3]), col=1 (X in [1, 2])
        cell = grid.get_cell(2, 1)
        assert cell["is_occupied"] is True
        assert cell["point_count"] == 2
        assert cell["min_z"] == pytest.approx(0.4)
        assert cell["max_z"] == pytest.approx(1.6)
        assert cell["mean_z"] == pytest.approx(1.0)
        assert cell["mean_intensity"] == pytest.approx(0.5)
        assert cell["elevation_diff"] == pytest.approx(1.2)


class TestUniform25DMapBuilder:
    """Test suite for map builder logic."""

    def test_xy_point_to_cell_assignment(self):
        pts = np.array([
            [0.5, 0.5, 1.0, 0.1],  # Cell (0, 0)
            [1.5, 0.5, 2.0, 0.2],  # Cell (0, 1)
            [0.5, 1.5, 3.0, 0.3],  # Cell (1, 0)
            [1.5, 1.5, 4.0, 0.4],  # Cell (1, 1)
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=2.0, min_y=0.0, max_y=2.0)
        )
        grid = builder.build_map(frame)

        assert grid.shape == (2, 2)
        assert grid.point_count[0, 0] == 1
        assert grid.point_count[0, 1] == 1
        assert grid.point_count[1, 0] == 1
        assert grid.point_count[1, 1] == 1
        assert grid.mean_z[0, 0] == pytest.approx(1.0)
        assert grid.mean_z[0, 1] == pytest.approx(2.0)
        assert grid.mean_z[1, 0] == pytest.approx(3.0)
        assert grid.mean_z[1, 1] == pytest.approx(4.0)

    def test_cell_statistics_computation(self):
        pts = np.array([
            [0.2, 0.2, 1.0, 0.1],
            [0.3, 0.4, 3.0, 0.5],
            [0.8, 0.8, 2.0, 0.3],
            [2.5, 2.5, 5.0, 0.9],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=3.0, min_y=0.0, max_y=3.0)
        )
        grid = builder.build_map(frame)

        # Cell (row 0, col 0) has 3 points
        assert grid.point_count[0, 0] == 3
        assert grid.min_z[0, 0] == pytest.approx(1.0)
        assert grid.max_z[0, 0] == pytest.approx(3.0)
        assert grid.mean_z[0, 0] == pytest.approx(2.0)
        assert grid.mean_intensity[0, 0] == pytest.approx(0.3)

        # Cell (row 2, col 2) has 1 point
        assert grid.point_count[2, 2] == 1
        assert grid.min_z[2, 2] == pytest.approx(5.0)

        # Empty cell (row 1, col 1)
        assert grid.point_count[1, 1] == 0
        assert np.isnan(grid.min_z[1, 1])
        assert np.isnan(grid.max_z[1, 1])
        assert np.isnan(grid.mean_z[1, 1])

    def test_grid_dimensions_for_various_resolutions(self):
        frame = LidarFrame(points=np.empty((0, 4), dtype=np.float32))
        
        # 10m x 20m with 0.5m res -> 40 rows x 20 cols
        cfg = MappingConfig(resolution=0.5, min_x=0.0, max_x=10.0, min_y=0.0, max_y=20.0)
        grid = Uniform25DMapBuilder(cfg).build_map(frame)
        assert grid.shape == (40, 20)
        assert grid.total_cells == 800

        # 10m x 20m with 2.0m res -> 10 rows x 5 cols
        cfg2 = MappingConfig(resolution=2.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=20.0)
        grid2 = Uniform25DMapBuilder(cfg2).build_map(frame)
        assert grid2.shape == (10, 5)
        assert grid2.total_cells == 50

    def test_points_outside_bounds_ignored(self):
        pts = np.array([
            [0.5, 0.5, 1.0, 0.5],
            [10.0, 10.0, 2.0, 0.5], # Outside
            [-5.0, 0.5, 3.0, 0.5],  # Outside
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=2.0, min_y=0.0, max_y=2.0)
        )
        grid = builder.build_map(frame)

        assert grid.num_occupied_cells == 1
        assert grid.metadata["input_points"] == 3
        assert grid.metadata["mapped_points"] == 1

    def test_boundary_conditions(self):
        # Point exactly on min bounds and on max bounds
        pts = np.array([
            [0.0, 0.0, 1.0, 0.5],
            [2.0, 2.0, 3.0, 0.5],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=2.0, min_y=0.0, max_y=2.0)
        )
        grid = builder.build_map(frame)

        assert grid.point_count[0, 0] >= 1
        assert grid.point_count[1, 1] >= 1
        assert grid.num_occupied_cells == 2

    def test_auto_bounds_behavior(self):
        pts = np.array([
            [2.0, 3.0, 0.5, 0.5],
            [6.0, 7.0, 1.5, 0.5],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        cfg = MappingConfig(resolution=1.0, auto_bounds=True)
        grid = Uniform25DMapBuilder(cfg).build_map(frame)

        assert grid.min_x <= 2.0
        assert grid.max_x >= 7.0
        assert grid.min_y <= 3.0
        assert grid.max_y >= 8.0
        assert grid.num_occupied_cells == 2

    def test_end_to_end_pipeline_integration(self):
        loader = SyntheticLidarLoader(num_frames=1, seed=42)
        raw_frame = loader[0]

        prep_pipeline = PreprocessingPipeline()
        clean_frame = prep_pipeline.process(raw_frame)

        map_builder = Uniform25DMapBuilder(
            MappingConfig(resolution=0.2, min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0)
        )
        grid = map_builder.build_map(clean_frame)

        assert grid.shape == (250, 250)
        assert grid.total_cells == 62500
        assert grid.num_occupied_cells > 0
        assert grid.num_empty_cells == 62500 - grid.num_occupied_cells
        assert grid.occupancy_percentage > 0.0
        assert grid.occupancy_percentage <= 100.0
