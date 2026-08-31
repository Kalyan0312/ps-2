"""
Unit and Integration Tests for Phase 13 Part A: Temporal Change Detection Between Adaptive Maps.

Tests:
1. First frame behavior (initial frame flag, newly observed assignment)
2. Identical consecutive maps (100% stable, 0 changes)
3. Small elevation differences below threshold (unchanged classification)
4. Elevation differences above threshold (changed classification, delta magnitude)
5. Newly observed regions (expansion of coverage)
6. No longer observed regions (contraction of coverage)
7. Different resolution tiers representing the same spatial region (common spatial reference comparison)
8. Boundary and corner coordinates (inclusive limits [-max, max])
9. Partially overlapping map bounds (union spatial domain)
10. TemporalMapManager state updates, history, and resets
"""

import pytest
import numpy as np
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.terrain.analyzer import TerrainAnalyzer
from backend.priority.selector import AdaptiveResolutionSelector
from backend.temporal.config import TemporalConfig, TemporalChangeConfig
from backend.temporal.result import (
    TemporalChangeResult,
    STATE_UNOCCUPIED,
    STATE_UNCHANGED,
    STATE_CHANGED,
    STATE_NEWLY_OBSERVED,
    STATE_NO_LONGER_OBSERVED,
)
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.temporal_manager import TemporalMapManager


# ---------------------------------------------------------------------------
# Helper Fixtures and Map Builders
# ---------------------------------------------------------------------------

@pytest.fixture
def base_mapping_config():
    return MappingConfig(
        resolution=0.5,
        min_x=-10.0,
        max_x=10.0,
        min_y=-10.0,
        max_y=10.0,
        auto_bounds=False,
    )


def build_adaptive_map_from_points(
    points: np.ndarray,
    map_cfg: MappingConfig,
    frame_id: int = 0,
) -> AdaptiveMap25D:
    """Helper to build an AdaptiveMap25D from point coordinates."""
    frame = LidarFrame(points=points, frame_id=frame_id, timestamp=float(frame_id))
    base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(frame)
    terrain = TerrainAnalyzer().analyze(base_grid)
    decision = AdaptiveResolutionSelector().select_resolution(base_grid, terrain)
    return AdaptiveMap25DBuilder(config=map_cfg).build(frame, decision)


# ---------------------------------------------------------------------------
# 1. First Frame Behavior
# ---------------------------------------------------------------------------

class TestFirstFrameBehavior:

    def test_initial_frame_detection(self, base_mapping_config):
        """Verifies initial frame has is_initial_frame=True, no false changes, and newly observed cells."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(num_ground_points=2000)
        amap = build_adaptive_map_from_points(frame.points, base_mapping_config, frame_id=0)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        result = detector.detect_change(current_grid=amap, previous_grid=None)

        assert result.is_initial_frame is True
        assert result.previous_frame_id is None
        assert result.current_frame_id == 0
        assert result.changed_cells == 0
        assert result.num_changed_cells == 0
        assert result.no_longer_observed_cells == 0
        assert result.newly_observed_cells > 0
        assert np.all(result.change_state[result.newly_observed_mask] == STATE_NEWLY_OBSERVED)
        assert np.all(result.change_state[result.unoccupied_mask] == STATE_UNOCCUPIED)
        assert result.change_percentage == 0.0


# ---------------------------------------------------------------------------
# 2. Identical Consecutive Maps
# ---------------------------------------------------------------------------

class TestIdenticalConsecutiveMaps:

    def test_identical_consecutive_adaptive_maps(self, base_mapping_config):
        """Verifies identical consecutive maps report 100% unchanged cells and zero changes."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(num_ground_points=2500)

        map0 = build_adaptive_map_from_points(frame.points, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(frame.points.copy(), base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        result = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert result.is_initial_frame is False
        assert result.previous_frame_id == 0
        assert result.current_frame_id == 1
        assert result.changed_cells == 0
        assert result.newly_observed_cells == 0
        assert result.no_longer_observed_cells == 0
        assert result.unchanged_cells == result.total_compared_cells
        assert result.unchanged_cells > 0
        assert result.change_percentage == 0.0
        assert result.stable_percentage == 100.0
        assert np.all(result.change_state[result.unchanged_mask] == STATE_UNCHANGED)


# ---------------------------------------------------------------------------
# 3 & 4. Elevation Differences Below and Above Threshold
# ---------------------------------------------------------------------------

class TestElevationChangeSensitivity:

    def test_elevation_difference_below_threshold(self, base_mapping_config):
        """Elevation delta (+0.08m) below threshold (0.20m) is classified as unchanged."""
        pts0 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [1.0, 1.0, 1.0, 100.0],
            [2.0, 2.0, 1.0, 100.0],
        ], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.08, 100.0],  # +0.08m
            [1.0, 1.0, 1.05, 100.0],  # +0.05m
            [2.0, 2.0, 1.02, 100.0],  # +0.02m
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.changed_cells == 0
        assert res.unchanged_cells >= 3
        assert res.num_elevation_changed_cells == 0

    def test_elevation_difference_above_threshold(self, base_mapping_config):
        """Elevation delta (+0.50m) above threshold (0.20m) is classified as changed."""
        pts0 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [1.0, 1.0, 1.0, 100.0],
            [2.0, 2.0, 1.0, 100.0],
        ], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.50, 100.0],  # +0.50m (CHANGED)
            [1.0, 1.0, 1.05, 100.0],  # +0.05m (UNCHANGED)
            [2.0, 2.0, 1.02, 100.0],  # +0.02m (UNCHANGED)
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.changed_cells >= 1
        assert res.num_elevation_changed_cells >= 1
        assert res.unchanged_cells >= 2

        # Verify summary stats
        summary = res.get_summary()
        assert summary["max_elevation_difference"] >= 0.45


# ---------------------------------------------------------------------------
# 5 & 6. Newly Observed and No Longer Observed Regions
# ---------------------------------------------------------------------------

class TestCoverageTransitions:

    def test_newly_observed_regions(self, base_mapping_config):
        """New points in previously empty region are classified as newly_observed."""
        pts0 = np.array([
            [0.0, 0.0, 1.0, 100.0],
        ], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [5.0, 5.0, 1.0, 100.0],  # New cell
            [6.0, 6.0, 1.0, 100.0],  # New cell
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.newly_observed_cells >= 2
        assert res.no_longer_observed_cells == 0
        assert res.unchanged_cells >= 1

    def test_no_longer_observed_regions(self, base_mapping_config):
        """Points absent from previous frame are classified as no_longer_observed."""
        pts0 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [5.0, 5.0, 1.0, 100.0],
            [6.0, 6.0, 1.0, 100.0],
        ], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            # (5, 5) and (6, 6) are missing
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.no_longer_observed_cells >= 2
        assert res.newly_observed_cells == 0
        assert res.unchanged_cells >= 1


# ---------------------------------------------------------------------------
# 7. Multi-Tier Spatial Comparison Across Resolutions
# ---------------------------------------------------------------------------

class TestMultiTierSpatialComparison:

    def test_different_resolution_tiers_in_same_region(self, base_mapping_config):
        """
        Verifies that comparing a coarse region with an ultra-fine region
        at the same physical location correctly identifies unchanged elevation.
        """
        # Frame 0: flat terrain (classified as coarse tier)
        pts0 = np.array([
            [x, y, 0.0, 100.0]
            for x in np.linspace(-3, 3, 20)
            for y in np.linspace(-3, 3, 20)
        ], dtype=np.float64)

        # Frame 1: dense points with small roughness (triggers fine/ultra_fine tier) at same 0.0m elevation
        pts1 = np.array([
            [x, y, 0.01 * np.sin(x), 100.0]
            for x in np.linspace(-3, 3, 40)
            for y in np.linspace(-3, 3, 40)
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        # Should match in common spatial domain with delta < 0.20m -> unchanged
        assert res.unchanged_cells > 0
        assert res.num_elevation_changed_cells == 0


# ---------------------------------------------------------------------------
# 8. Boundary and Corner Coordinates
# ---------------------------------------------------------------------------

class TestBoundaryAndCornerCoordinates:

    def test_inclusive_corner_points_temporal_comparison(self, base_mapping_config):
        """Corner points on exact min/max bounds must not trigger indexing errors."""
        corners0 = np.array([
            [-10.0, -10.0, 1.0, 100.0],
            [ 10.0, -10.0, 1.0, 100.0],
            [-10.0,  10.0, 1.0, 100.0],
            [ 10.0,  10.0, 1.0, 100.0],
        ], dtype=np.float64)

        corners1 = np.array([
            [-10.0, -10.0, 1.0, 100.0],  # stable
            [ 10.0, -10.0, 2.5, 100.0],  # changed elevation (+1.5m)
            [-10.0,  10.0, 1.0, 100.0],  # stable
            # [10, 10] removed
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(corners0, base_mapping_config, frame_id=0)
        map1 = build_adaptive_map_from_points(corners1, base_mapping_config, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.grid_shape == map1.base_shape
        assert res.no_longer_observed_cells >= 1
        assert res.changed_cells >= 1


# ---------------------------------------------------------------------------
# 9. Partially Overlapping Map Bounds
# ---------------------------------------------------------------------------

class TestPartiallyOverlappingBounds:

    def test_partially_overlapping_maps(self):
        """Maps with different spatial extents compare correctly across the union domain."""
        cfg0 = MappingConfig(resolution=0.5, min_x=-10.0, max_x=5.0, min_y=-10.0, max_y=5.0, auto_bounds=False)
        cfg1 = MappingConfig(resolution=0.5, min_x=-5.0, max_x=10.0, min_y=-5.0, max_y=10.0, auto_bounds=False)

        pts0 = np.array([
            [-2.0, -2.0, 1.0, 100.0],  # in overlap
            [-8.0, -8.0, 1.0, 100.0],  # map0 only
        ], dtype=np.float64)

        pts1 = np.array([
            [-2.0, -2.0, 1.0, 100.0],  # in overlap (stable)
            [ 8.0,  8.0, 1.0, 100.0],  # map1 only (new)
        ], dtype=np.float64)

        map0 = build_adaptive_map_from_points(pts0, cfg0, frame_id=0)
        map1 = build_adaptive_map_from_points(pts1, cfg1, frame_id=1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        res = detector.detect_change(current_grid=map1, previous_grid=map0)

        assert res.bounds == (-10.0, 10.0, -10.0, 10.0)
        assert res.unchanged_cells >= 1
        assert res.newly_observed_cells >= 1
        assert res.no_longer_observed_cells >= 1


# ---------------------------------------------------------------------------
# 10. TemporalMapManager State Updates
# ---------------------------------------------------------------------------

class TestTemporalMapManager:

    def test_temporal_map_manager_lifecycle(self, base_mapping_config):
        """Tests TemporalMapManager sequence updates, history recording, and reset."""
        manager = TemporalMapManager(TemporalChangeConfig(elevation_change_threshold=0.20))
        assert manager.has_history is False
        assert manager.frame_count == 0

        # Frame 0: Baseline
        pts0 = np.array([[0.0, 0.0, 1.0, 100.0]], dtype=np.float64)
        map0 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=0)
        r0 = manager.update(map0)
        assert r0.is_initial_frame is True
        assert manager.has_history is True
        assert manager.frame_count == 1
        assert manager.last_map is map0

        # Frame 1: Identical
        map1 = build_adaptive_map_from_points(pts0, base_mapping_config, frame_id=1)
        r1 = manager.update(map1)
        assert r1.is_initial_frame is False
        assert r1.changed_cells == 0
        assert r1.unchanged_cells >= 1
        assert manager.frame_count == 2

        # Frame 2: Elevation change
        pts2 = np.array([[0.0, 0.0, 2.0, 100.0]], dtype=np.float64)
        map2 = build_adaptive_map_from_points(pts2, base_mapping_config, frame_id=2)
        r2 = manager.update(map2)
        assert r2.changed_cells >= 1
        assert manager.frame_count == 3
        assert len(manager.history) == 3

        # Summary
        summary = manager.get_summary()
        assert summary["frame_count"] == 3
        assert summary["has_history"] is True
        assert summary["history_length"] == 3

        # Reset
        manager.reset()
        assert manager.has_history is False
        assert manager.frame_count == 0
        assert len(manager.history) == 0
