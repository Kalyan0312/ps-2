"""
Unit and Integration Tests for Phase 9: Temporal Multi-Frame Processing and Change Detection.
"""

import pytest
import numpy as np
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.mapping.grid_map import GridMap25D
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.mapping.config import MappingConfig
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig

from backend.temporal.config import TemporalConfig
from backend.temporal.result import TemporalChangeResult
from backend.temporal.state import TemporalState
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.processor import TemporalProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_test_grid(
    occupancy_pattern: np.ndarray,
    elevation_pattern: np.ndarray,
    resolution: float = 1.0,
    min_x: float = 0.0,
    max_x: float = 5.0,
    min_y: float = 0.0,
    max_y: float = 5.0,
    frame_id: int = 0,
) -> GridMap25D:
    """Helper to create a small 5x5 GridMap25D with deterministic patterns."""
    rows, cols = occupancy_pattern.shape
    point_count = np.where(occupancy_pattern, 10, 0).astype(np.int64)
    mean_z = np.where(occupancy_pattern, elevation_pattern, np.nan).astype(np.float64)
    min_z = mean_z.copy()
    max_z = mean_z.copy()
    mean_intensity = np.where(occupancy_pattern, 0.5, np.nan).astype(np.float64)

    return GridMap25D(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        resolution=resolution,
        point_count=point_count,
        min_z=min_z,
        max_z=max_z,
        mean_z=mean_z,
        mean_intensity=mean_intensity,
        metadata={"frame_id": frame_id},
    )


@pytest.fixture
def base_occ():
    occ = np.zeros((5, 5), dtype=bool)
    occ[1:4, 1:4] = True  # 3x3 occupied center
    return occ


@pytest.fixture
def base_elev():
    elev = np.zeros((5, 5), dtype=np.float32)
    elev[1:4, 1:4] = 1.0
    return elev


# ---------------------------------------------------------------------------
# Test TemporalConfig
# ---------------------------------------------------------------------------

class TestTemporalConfig:

    def test_default_config(self):
        cfg = TemporalConfig()
        assert cfg.enabled is True
        assert cfg.elevation_change_threshold == 0.15
        assert cfg.min_points_per_cell == 1
        assert cfg.reuse_stable_regions is True

    def test_invalid_values_raise_error(self):
        with pytest.raises(ValueError, match="elevation_change_threshold must be non-negative"):
            TemporalConfig(elevation_change_threshold=-0.1)

        with pytest.raises(ValueError, match="min_points_per_cell must be >= 1"):
            TemporalConfig(min_points_per_cell=0)

    def test_from_yaml(self):
        config_path = Path("configs/config.yaml")
        if config_path.is_file():
            cfg = TemporalConfig.from_yaml(config_path)
            assert cfg.enabled is True
            assert cfg.elevation_change_threshold >= 0.0


# ---------------------------------------------------------------------------
# Test TemporalChangeDetector
# ---------------------------------------------------------------------------

class TestTemporalChangeDetector:

    def test_initial_frame_detection(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)
        detector = TemporalChangeDetector(TemporalConfig(elevation_change_threshold=0.10))

        result = detector.detect_change(current_grid=grid0, previous_grid=None)

        assert result.is_initial_frame is True
        assert result.num_changed_cells == 0
        assert result.num_stable_cells == 9
        assert result.num_new_occupied_cells == 0
        assert result.num_removed_occupied_cells == 0
        assert result.num_elevation_changed_cells == 0
        assert result.change_percentage == 0.0
        assert result.stable_percentage == 100.0

    def test_identical_consecutive_frames(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)
        grid1 = make_test_grid(base_occ, base_elev, frame_id=1)
        detector = TemporalChangeDetector(TemporalConfig(elevation_change_threshold=0.10))

        result = detector.detect_change(current_grid=grid1, previous_grid=grid0)

        assert result.is_initial_frame is False
        assert result.num_changed_cells == 0
        assert result.num_stable_cells == 9
        assert result.change_percentage == 0.0
        assert result.stable_percentage == 100.0

    def test_new_occupancy_detection(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)

        # Add 2 new cells at (0, 0) and (0, 1)
        new_occ = base_occ.copy()
        new_occ[0, 0] = True
        new_occ[0, 1] = True
        new_elev = base_elev.copy()
        new_elev[0, 0] = 2.0
        new_elev[0, 1] = 2.0

        grid1 = make_test_grid(new_occ, new_elev, frame_id=1)
        detector = TemporalChangeDetector(TemporalConfig())

        result = detector.detect_change(current_grid=grid1, previous_grid=grid0)

        assert result.num_new_occupied_cells == 2
        assert result.num_removed_occupied_cells == 0
        assert result.num_elevation_changed_cells == 0
        assert result.num_changed_cells == 2
        assert result.num_stable_cells == 9
        assert result.new_occupancy_mask[0, 0] is True or result.new_occupancy_mask[0, 0] == 1
        assert result.new_occupancy_mask[0, 1] is True or result.new_occupancy_mask[0, 1] == 1

    def test_removed_occupancy_detection(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)

        # Remove 1 cell at (2, 2)
        rem_occ = base_occ.copy()
        rem_occ[2, 2] = False
        rem_elev = base_elev.copy()

        grid1 = make_test_grid(rem_occ, rem_elev, frame_id=1)
        detector = TemporalChangeDetector(TemporalConfig())

        result = detector.detect_change(current_grid=grid1, previous_grid=grid0)

        assert result.num_new_occupied_cells == 0
        assert result.num_removed_occupied_cells == 1
        assert result.num_elevation_changed_cells == 0
        assert result.num_changed_cells == 1
        assert result.num_stable_cells == 8
        assert result.removed_occupancy_mask[2, 2] is True or result.removed_occupancy_mask[2, 2] == 1

    def test_elevation_change_detection(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)

        # Modify elevation at (2, 2) from 1.0 to 1.5 (+0.5m)
        mod_elev = base_elev.copy()
        mod_elev[2, 2] = 1.5

        grid1 = make_test_grid(base_occ, mod_elev, frame_id=1)
        detector = TemporalChangeDetector(TemporalConfig(elevation_change_threshold=0.20))

        result = detector.detect_change(current_grid=grid1, previous_grid=grid0)

        assert result.num_new_occupied_cells == 0
        assert result.num_removed_occupied_cells == 0
        assert result.num_elevation_changed_cells == 1
        assert result.num_changed_cells == 1
        assert result.num_stable_cells == 8
        assert result.elevation_changed_mask[2, 2] is True or result.elevation_changed_mask[2, 2] == 1
        assert pytest.approx(result.elevation_difference[2, 2], 0.01) == 0.5

    def test_elevation_threshold_sensitivity(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)

        # Modify elevation by +0.10m
        mod_elev = base_elev.copy()
        mod_elev[2, 2] = 1.10
        grid1 = make_test_grid(base_occ, mod_elev, frame_id=1)

        # Strict threshold 0.05m -> flagged as changed
        det_strict = TemporalChangeDetector(TemporalConfig(elevation_change_threshold=0.05))
        res_strict = det_strict.detect_change(grid1, grid0)
        assert res_strict.num_elevation_changed_cells == 1

        # Relaxed threshold 0.20m -> considered stable
        det_relaxed = TemporalChangeDetector(TemporalConfig(elevation_change_threshold=0.20))
        res_relaxed = det_relaxed.detect_change(grid1, grid0)
        assert res_relaxed.num_elevation_changed_cells == 0
        assert res_relaxed.num_stable_cells == 9

    def test_incompatible_grids_raise_error(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, resolution=1.0, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0)
        # Different resolution & shape (10x10 for 5.0m span @ 0.5m)
        occ10 = np.zeros((10, 10), dtype=bool)
        elev10 = np.zeros((10, 10), dtype=np.float32)
        grid_bad_res = make_test_grid(occ10, elev10, resolution=0.5, min_x=0.0, max_x=5.0, min_y=0.0, max_y=5.0)
        # Different bounds (5x5 for 5.0m span @ 1.0m)
        grid_bad_bounds = make_test_grid(base_occ, base_elev, resolution=1.0, min_x=-5.0, max_x=0.0, min_y=0.0, max_y=5.0)

        detector = TemporalChangeDetector()
        with pytest.raises(ValueError, match="Incompatible grid resolutions|Incompatible grid shapes"):
            detector.detect_change(grid_bad_res, grid0)

        with pytest.raises(ValueError, match="Incompatible grid bounds"):
            detector.detect_change(grid_bad_bounds, grid0)

    def test_cell_query_interface(self, base_occ, base_elev):
        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)

        new_occ = base_occ.copy()
        new_occ[0, 0] = True
        new_elev = base_elev.copy()
        new_elev[0, 0] = 2.0
        grid1 = make_test_grid(new_occ, new_elev, frame_id=1)

        detector = TemporalChangeDetector()
        res = detector.detect_change(grid1, grid0)

        cell_new = res.get_cell(0, 0)
        assert cell_new["status"] == "new_occupancy"
        assert cell_new["is_changed"] is True
        assert cell_new["is_new_occupancy"] is True

        cell_stable = res.get_cell(2, 2)
        assert cell_stable["status"] == "stable"
        assert cell_stable["is_stable"] is True

        cell_unocc = res.get_cell(4, 4)
        assert cell_unocc["status"] == "unoccupied"
        assert cell_unocc["is_changed"] is False

        with pytest.raises(IndexError):
            res.get_cell(10, 10)


# ---------------------------------------------------------------------------
# Test TemporalState and TemporalProcessor
# ---------------------------------------------------------------------------

class TestTemporalProcessor:

    def test_processor_multi_frame_sequence(self, base_occ, base_elev):
        processor = TemporalProcessor(TemporalConfig(elevation_change_threshold=0.15))

        grid0 = make_test_grid(base_occ, base_elev, frame_id=0)
        grid1 = make_test_grid(base_occ, base_elev, frame_id=1)

        # Frame 0
        r0 = processor.process_grid_map(grid0)
        assert r0.is_initial_frame is True
        assert processor.state.frame_count == 1
        assert processor.state.has_history is True

        # Frame 1 (Identical)
        r1 = processor.process_grid_map(grid1)
        assert r1.is_initial_frame is False
        assert r1.num_changed_cells == 0
        assert r1.stable_percentage == 100.0
        assert processor.state.frame_count == 2

        # Frame 2 (New obstacle + removed cell)
        mod_occ = base_occ.copy()
        mod_occ[0, 0] = True
        mod_occ[1, 1] = False
        grid2 = make_test_grid(mod_occ, base_elev, frame_id=2)

        r2 = processor.process_grid_map(grid2)
        assert r2.num_new_occupied_cells == 1
        assert r2.num_removed_occupied_cells == 1
        assert r2.num_changed_cells == 2
        assert processor.state.frame_count == 3
        assert len(processor.state.history) == 3

        # Reset
        processor.reset()
        assert processor.state.frame_count == 0
        assert processor.state.has_history is False

    def test_synthetic_sequence_generator_scenarios(self):
        gen = SyntheticLidarGenerator(seed=42)

        # Static scenario
        frames_static = gen.generate_temporal_sequence("static")
        assert len(frames_static) == 2
        np.testing.assert_allclose(frames_static[0].points, frames_static[1].points)

        # New obstacle scenario
        frames_new = gen.generate_temporal_sequence("new_obstacle")
        assert len(frames_new) == 2
        assert frames_new[1].num_points > frames_new[0].num_points

        # Removed obstacle scenario
        frames_rem = gen.generate_temporal_sequence("removed_obstacle")
        assert len(frames_rem) == 2
        assert frames_rem[1].num_points < frames_rem[0].num_points

        # Elevation change scenario
        frames_elev = gen.generate_temporal_sequence("elevation_change")
        assert len(frames_elev) == 2

        # Dynamic 4-frame sequence
        frames_dyn = gen.generate_temporal_sequence("dynamic")
        assert len(frames_dyn) == 4


# ---------------------------------------------------------------------------
# Test End-to-End Pipeline Integration with Temporal Processing
# ---------------------------------------------------------------------------

class TestTemporalPipelineIntegration:

    def test_pipeline_through_temporal_processor(self):
        gen = SyntheticLidarGenerator(seed=42)
        frames = gen.generate_temporal_sequence("dynamic")

        map_cfg = MappingConfig(resolution=0.5, min_x=-20.0, max_x=20.0, min_y=-20.0, max_y=20.0)
        prep_cfg = PreprocessingConfig()
        temporal_cfg = TemporalConfig(elevation_change_threshold=0.15)

        preprocessor = PreprocessingPipeline(config=prep_cfg)
        builder = Uniform25DMapBuilder(config=map_cfg)
        processor = TemporalProcessor(config=temporal_cfg)

        results = processor.process_sequence(
            frames=frames,
            builder=builder,
            preprocessor=preprocessor,
        )

        assert len(results) == 4
        # Frame 0: initial baseline
        assert results[0].is_initial_frame is True
        # Frame 1: static repeat
        assert results[1].num_changed_cells == 0
        assert results[1].stable_percentage == 100.0
        # Frame 2: dynamic multi-change
        assert results[2].num_changed_cells > 0
        assert results[2].num_new_occupied_cells > 0
        assert results[2].num_removed_occupied_cells > 0
        # Frame 3: stabilized
        assert results[3].num_changed_cells == 0
        assert results[3].stable_percentage == 100.0
