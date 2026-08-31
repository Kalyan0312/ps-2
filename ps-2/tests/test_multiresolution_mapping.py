"""
Unit tests for Multi-Resolution 2.5D Elevation Grid Mapping Foundation (Phase 3B).
"""

import numpy as np
import pytest
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarLoader
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.mapping.grid_map import GridMap25D
from backend.mapping.config import MappingConfig, DEFAULT_MULTIRES_LEVELS
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.multires_map import MultiResolutionMap
from backend.mapping.multires_builder import MultiResolution25DMapBuilder


class TestMultiResolutionMap:
    """Test suite for MultiResolutionMap data structure."""

    def test_multires_map_initialization_and_access(self):
        # Create mock 2.5D grid maps for 0.50m and 0.25m
        map_coarse = Uniform25DMapBuilder(
            MappingConfig(resolution=0.50, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        map_medium = Uniform25DMapBuilder(
            MappingConfig(resolution=0.25, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        multires = MultiResolutionMap(
            maps={0.50: map_coarse, 0.25: map_medium},
            levels={"coarse": 0.50, "medium": 0.25},
        )

        assert len(multires) == 2
        assert multires.available_resolutions == [0.50, 0.25]
        assert multires.available_levels == ["coarse", "medium"]

        # Access by float
        assert multires.get_map(0.50) is map_coarse
        assert multires.get_map(0.25) is map_medium
        assert multires[0.50] is map_coarse

        # Access by string alias
        assert multires.get_map("coarse") is map_coarse
        assert multires.get_map("medium") is map_medium
        assert multires["coarse"] is map_coarse

        # Access by numeric string
        assert multires.get_map("0.50") is map_coarse

    def test_invalid_resolution_raises_key_error(self):
        multires = MultiResolutionMap(
            maps={0.50: Uniform25DMapBuilder(MappingConfig(resolution=0.50)).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))},
            levels={"coarse": 0.50},
        )

        with pytest.raises(KeyError):
            multires.get_map(0.10)

        with pytest.raises(KeyError):
            multires.get_map("ultra_fine")

        with pytest.raises(KeyError):
            multires.get_map("non_existent_level")

    def test_get_summary(self):
        pts = np.array([[1.0, 1.0, 0.5, 0.8]], dtype=np.float32)
        frame = LidarFrame(points=pts)
        builder = MultiResolution25DMapBuilder(
            MappingConfig(
                min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0,
                multi_resolution_levels={"coarse": 0.50, "medium": 0.25}
            )
        )
        multires = builder.build_multires_map(frame)
        summary = multires.get_summary()

        assert len(summary) == 2
        assert summary[0]["level_name"] == "coarse"
        assert summary[0]["resolution"] == 0.50
        assert summary[0]["occupied_cells"] == 1
        assert summary[1]["level_name"] == "medium"
        assert summary[1]["resolution"] == 0.25
        assert summary[1]["occupied_cells"] == 1


class TestMultiResolution25DMapBuilder:
    """Test suite for multi-scale map construction."""

    def test_all_configured_resolution_levels_built(self):
        pts = np.array([
            [2.0, 3.0, 0.5, 0.4],
            [10.0, 12.0, 1.8, 0.7],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        cfg = MappingConfig(
            min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0,
            multi_resolution_levels={
                "coarse": 0.50,
                "medium": 0.25,
                "fine": 0.10,
                "ultra_fine": 0.05,
            }
        )
        builder = MultiResolution25DMapBuilder(cfg)
        multires = builder.build_multires_map(frame)

        assert len(multires) == 4
        assert multires.available_resolutions == [0.50, 0.25, 0.10, 0.05]

        # Verify grid dimensions at different resolutions for 50m x 50m span
        assert multires["coarse"].shape == (100, 100)
        assert multires["coarse"].total_cells == 10000

        assert multires["medium"].shape == (200, 200)
        assert multires["medium"].total_cells == 40000

        assert multires["fine"].shape == (500, 500)
        assert multires["fine"].total_cells == 250000

        assert multires["ultra_fine"].shape == (1000, 1000)
        assert multires["ultra_fine"].total_cells == 1000000

    def test_data_consistency_across_scales(self):
        pts = np.array([
            [5.0, 5.0, 2.0, 0.5],
            [5.1, 5.1, 4.0, 0.9],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts)

        cfg = MappingConfig(
            min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0,
            multi_resolution_levels={"coarse": 1.0, "fine": 0.05}
        )
        builder = MultiResolution25DMapBuilder(cfg)
        multires = builder.build_multires_map(frame)

        coarse_map = multires["coarse"]
        fine_map = multires["fine"]

        # Coarse map: both points fall in single 1.0m cell (row 5, col 5)
        cell_coarse = coarse_map.get_cell(5, 5)
        assert cell_coarse["point_count"] == 2
        assert cell_coarse["min_z"] == pytest.approx(2.0)
        assert cell_coarse["max_z"] == pytest.approx(4.0)
        assert cell_coarse["mean_z"] == pytest.approx(3.0)

        # Fine map (0.05m): points fall in two distinct 0.05m cells
        assert fine_map.num_occupied_cells == 2

    def test_yaml_config_loading(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        cfg = MappingConfig.from_yaml(config_path)
        
        assert "coarse" in cfg.multi_resolution_levels
        assert "ultra_fine" in cfg.multi_resolution_levels
        assert cfg.multi_resolution_levels["coarse"] == pytest.approx(0.50)
        assert cfg.multi_resolution_levels["ultra_fine"] == pytest.approx(0.05)

    def test_end_to_end_multires_pipeline(self):
        loader = SyntheticLidarLoader(num_frames=1, seed=42)
        raw_frame = loader[0]

        prep_pipeline = PreprocessingPipeline()
        clean_frame = prep_pipeline.process(raw_frame)

        builder = MultiResolution25DMapBuilder(
            MappingConfig(
                min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0,
                multi_resolution_levels={
                    "coarse": 0.50,
                    "medium": 0.25,
                    "fine": 0.10,
                    "ultra_fine": 0.05,
                }
            )
        )
        multires = builder.build_multires_map(clean_frame)

        for res in multires.available_resolutions:
            grid = multires.get_map(res)
            assert grid.num_occupied_cells > 0
            assert grid.occupancy_percentage > 0.0
            assert grid.total_cells > 0


class TestPhase3ACompatibility:
    """Verify Phase 3A Uniform25DMapBuilder remains 100% compatible."""

    def test_uniform_builder_still_functional(self):
        pts = np.array([[0.5, 0.5, 1.0, 0.5]], dtype=np.float32)
        frame = LidarFrame(points=pts)

        cfg = MappingConfig(resolution=0.20, min_x=-5.0, max_x=5.0, min_y=-5.0, max_y=5.0)
        uniform_builder = Uniform25DMapBuilder(cfg)
        grid = uniform_builder.build_map(frame)

        assert isinstance(grid, GridMap25D)
        assert grid.resolution == pytest.approx(0.20)
        assert grid.shape == (50, 50)
        assert grid.num_occupied_cells == 1
