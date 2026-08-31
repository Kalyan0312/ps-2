"""
Unit and Integration Tests for Phase 13 Part B: Safe Incremental Adaptive Map Updates.

Tests:
1. Identical consecutive maps (100% reuse)
2. Mostly stable scene (high reuse, small selective rebuild)
3. Changed elevation regions (rebuilds affected cells)
4. Newly observed regions (adds new spatial representation)
5. No longer observed regions (invalidates old cells)
6. Tier assignment changes (forces rebuild when tier changes)
7. Different resolution tiers in active map
8. Inclusive boundary corner coordinates
9. Partially overlapping spatial bounds
10. Incremental result equals full rebuild (point_count, min_z, max_z, mean_z, mean_intensity)
11. world_query() equivalence between incremental and full rebuild
12. TemporalMapManager integration
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
from backend.temporal.config import TemporalChangeConfig
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.temporal_manager import TemporalMapManager
from backend.temporal.incremental_builder import (
    IncrementalAdaptiveMapBuilder,
    IncrementalBuildResult,
)


# ---------------------------------------------------------------------------
# Fixtures & Helper Functions
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


def run_pipeline(frame: LidarFrame, map_cfg: MappingConfig):
    """Executes base mapping, terrain analysis, and resolution selection."""
    base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(frame)
    terrain = TerrainAnalyzer().analyze(base_grid)
    decision = AdaptiveResolutionSelector().select_resolution(base_grid, terrain)
    return base_grid, terrain, decision


# ---------------------------------------------------------------------------
# 1. Identical Consecutive Maps
# ---------------------------------------------------------------------------

class TestIdenticalConsecutiveMaps:

    def test_identical_consecutive_incremental_build(self, base_mapping_config):
        """Identical consecutive frames achieve near 100% reuse and exact full rebuild equivalence."""
        gen = SyntheticLidarGenerator(seed=42)
        frame0 = gen.generate_frame(frame_id=0, num_ground_points=2500)
        frame1 = LidarFrame(points=frame0.points.copy(), frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(frame0, base_mapping_config)
        full_builder = AdaptiveMap25DBuilder(config=base_mapping_config)
        map0 = full_builder.build(frame0, dec0)

        _, _, dec1 = run_pipeline(frame1, base_mapping_config)
        full_map1 = full_builder.build(frame1, dec1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        temp_res = detector.detect_change(map0, map0)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(
            previous_map=map0,
            current_frame=frame1,
            current_decision=dec1,
            temporal_result=temp_res,
        )

        assert inc_res.reused_cells > 0
        assert inc_res.rebuilt_cells == 0
        assert inc_res.reused_ratio == 1.0
        assert inc_res.adaptive_map.num_represented_cells == full_map1.num_represented_cells

        # Check world_query equivalence
        for tier in inc_res.adaptive_map.available_tiers:
            tg_inc = inc_res.adaptive_map.tier_maps[tier]
            tg_full = full_map1.tier_maps[tier]
            assert np.array_equal(tg_inc.point_count, tg_full.point_count)


# ---------------------------------------------------------------------------
# 2 & 3. Mostly Stable Scene and Changed Elevation Regions
# ---------------------------------------------------------------------------

class TestStableAndChangedScenes:

    def test_mostly_stable_scene_selective_rebuild(self, base_mapping_config):
        """Modifying 10% of cells selectively rebuilds only those changed regions."""
        gen = SyntheticLidarGenerator(seed=42)
        frame0 = gen.generate_frame(frame_id=0, num_ground_points=3000)

        pts1 = frame0.points.copy()
        # Elevate central radius by +0.75m
        center_mask = np.hypot(pts1[:, 0], pts1[:, 1]) < 2.0
        pts1[center_mask, 2] += 0.75
        frame1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(frame0, base_mapping_config)
        full_builder = AdaptiveMap25DBuilder(config=base_mapping_config)
        map0 = full_builder.build(frame0, dec0)

        _, _, dec1 = run_pipeline(frame1, base_mapping_config)
        full_map1 = full_builder.build(frame1, dec1)

        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        # Compare base grids
        temp_res = detector.detect_change(dec1.grid_map, dec0.grid_map)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, frame1, dec1, temp_res)

        assert inc_res.reused_cells > 0
        assert inc_res.rebuilt_cells > 0
        assert inc_res.reused_ratio > 0.50
        assert inc_res.adaptive_map.num_represented_cells > 0


# ---------------------------------------------------------------------------
# 4 & 5. Newly Observed and No Longer Observed Regions
# ---------------------------------------------------------------------------

class TestNewlyAndNoLongerObserved:

    def test_newly_observed_regions_added(self, base_mapping_config):
        """Newly observed points are incorporated into the incremental map."""
        pts0 = np.array([[0.0, 0.0, 1.0, 100.0]], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [5.0, 5.0, 1.0, 100.0],
        ], dtype=np.float64)

        f0 = LidarFrame(points=pts0, frame_id=0, timestamp=0.0)
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(f0, base_mapping_config)
        map0 = AdaptiveMap25DBuilder(config=base_mapping_config).build(f0, dec0)

        _, _, dec1 = run_pipeline(f1, base_mapping_config)
        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        temp_res = detector.detect_change(dec1.grid_map, dec0.grid_map)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, f1, dec1, temp_res)

        # Query the newly observed location
        q = inc_res.adaptive_map.world_query(5.0, 5.0)
        assert q.is_represented is True
        assert abs(q.mean_z - 1.0) < 1e-4

    def test_no_longer_observed_regions_invalidated(self, base_mapping_config):
        """Cleared locations in the current frame become unrepresented in the incremental map."""
        pts0 = np.array([
            [0.0, 0.0, 1.0, 100.0],
            [5.0, 5.0, 1.0, 100.0],
        ], dtype=np.float64)
        pts1 = np.array([
            [0.0, 0.0, 1.0, 100.0],
        ], dtype=np.float64)

        f0 = LidarFrame(points=pts0, frame_id=0, timestamp=0.0)
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(f0, base_mapping_config)
        map0 = AdaptiveMap25DBuilder(config=base_mapping_config).build(f0, dec0)

        _, _, dec1 = run_pipeline(f1, base_mapping_config)
        detector = TemporalChangeDetector(TemporalChangeConfig(elevation_change_threshold=0.20))
        temp_res = detector.detect_change(dec1.grid_map, dec0.grid_map)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, f1, dec1, temp_res)

        assert inc_res.invalidated_cells >= 1
        q = inc_res.adaptive_map.world_query(5.0, 5.0)
        assert q.is_represented is False


# ---------------------------------------------------------------------------
# 6 & 7. Tier Assignment Changes and Multi-Resolution Tiers
# ---------------------------------------------------------------------------

class TestTierAssignmentChanges:

    def test_tier_change_forces_rebuild(self, base_mapping_config):
        """
        When a region's assigned resolution tier changes (e.g. coarse -> fine),
        the cell must NOT be reused and must be rebuilt at the new resolution.
        """
        # Frame 0: flat terrain at (0, 0)
        pts0 = np.array([[x, y, 0.0, 100.0] for x in np.linspace(-2, 2, 10) for y in np.linspace(-2, 2, 10)], dtype=np.float64)
        # Frame 1: rough terrain at (0, 0) (forces fine/ultra_fine tier)
        pts1 = np.array([[x, y, 0.5 * (x**2 + y**2), 100.0] for x in np.linspace(-2, 2, 20) for y in np.linspace(-2, 2, 20)], dtype=np.float64)

        f0 = LidarFrame(points=pts0, frame_id=0, timestamp=0.0)
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(f0, base_mapping_config)
        map0 = AdaptiveMap25DBuilder(config=base_mapping_config).build(f0, dec0)

        _, _, dec1 = run_pipeline(f1, base_mapping_config)
        full_map1 = AdaptiveMap25DBuilder(config=base_mapping_config).build(f1, dec1)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, f1, dec1)

        # Ensure tier reassignment cells were marked for rebuild
        assert inc_res.rebuilt_cells > 0
        assert inc_res.adaptive_map.num_represented_cells == full_map1.num_represented_cells


# ---------------------------------------------------------------------------
# 8 & 9. Boundary Conditions and Partially Overlapping Bounds
# ---------------------------------------------------------------------------

class TestBoundaryAndOverlappingBounds:

    def test_boundary_corner_coordinates(self, base_mapping_config):
        """Exact boundary corners [-10, 10] are handled safely without index errors."""
        corners = np.array([
            [-10.0, -10.0, 1.0, 100.0],
            [ 10.0, -10.0, 2.0, 100.0],
            [-10.0,  10.0, 3.0, 100.0],
            [ 10.0,  10.0, 4.0, 100.0],
        ], dtype=np.float64)
        frame = LidarFrame(points=corners, frame_id=0, timestamp=0.0)

        _, _, dec = run_pipeline(frame, base_mapping_config)
        map0 = AdaptiveMap25DBuilder(config=base_mapping_config).build(frame, dec)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, frame, dec)

        assert inc_res.reused_ratio == 1.0
        assert inc_res.adaptive_map.num_represented_cells == map0.num_represented_cells

    def test_partially_overlapping_bounds(self):
        """Maps with different spatial extents handle incremental rebuilds safely."""
        cfg0 = MappingConfig(resolution=0.5, min_x=-10.0, max_x=5.0, min_y=-10.0, max_y=5.0, auto_bounds=False)
        cfg1 = MappingConfig(resolution=0.5, min_x=-5.0, max_x=10.0, min_y=-5.0, max_y=10.0, auto_bounds=False)

        pts0 = np.array([[-2.0, -2.0, 1.0, 100.0]], dtype=np.float64)
        pts1 = np.array([[-2.0, -2.0, 1.0, 100.0], [8.0, 8.0, 1.0, 100.0]], dtype=np.float64)

        f0 = LidarFrame(points=pts0, frame_id=0, timestamp=0.0)
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(f0, cfg0)
        map0 = AdaptiveMap25DBuilder(config=cfg0).build(f0, dec0)

        _, _, dec1 = run_pipeline(f1, cfg1)
        inc_builder = IncrementalAdaptiveMapBuilder(config=cfg1)
        inc_res = inc_builder.build(map0, f1, dec1)

        assert inc_res.adaptive_map.num_represented_cells >= 2

    def test_all_multi_resolution_tiers_in_incremental_map(self, base_mapping_config):
        """Maps with coarse, medium, fine, and ultra_fine tiers increment correctly."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(frame_id=0, num_ground_points=6000)

        _, _, dec = run_pipeline(frame, base_mapping_config)
        full_map = AdaptiveMap25DBuilder(config=base_mapping_config).build(frame, dec)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(full_map, frame, dec)

        assert len(inc_res.adaptive_map.tier_maps) == len(full_map.tier_maps)
        assert inc_res.reused_ratio == 1.0


# ---------------------------------------------------------------------------
# 10 & 11. Full Rebuild and world_query Equivalence
# ---------------------------------------------------------------------------

class TestFullRebuildEquivalence:

    def test_incremental_equals_full_rebuild_grid_layers(self, base_mapping_config):
        """
        Validates that an incremental update with 100% reuse produces identical
        point_count, min_z, max_z, mean_z, and mean_intensity across all tiers.
        """
        gen = SyntheticLidarGenerator(seed=99)
        frame0 = gen.generate_frame(frame_id=0, num_ground_points=3500)
        frame1 = LidarFrame(points=frame0.points.copy(), frame_id=1, timestamp=0.1)

        _, _, dec0 = run_pipeline(frame0, base_mapping_config)
        map0 = AdaptiveMap25DBuilder(config=base_mapping_config).build(frame0, dec0)

        _, _, dec1 = run_pipeline(frame1, base_mapping_config)
        full_map1 = AdaptiveMap25DBuilder(config=base_mapping_config).build(frame1, dec1)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(map0, frame1, dec1)
        inc_map = inc_res.adaptive_map

        assert inc_map.available_tiers == full_map1.available_tiers

        for tier in inc_map.available_tiers:
            tg_inc = inc_map.tier_maps[tier]
            tg_full = full_map1.tier_maps[tier]

            assert np.array_equal(tg_inc.point_count, tg_full.point_count)
            assert np.array_equal(np.isnan(tg_inc.min_z), np.isnan(tg_full.min_z))
            occ = ~np.isnan(tg_full.min_z)
            assert np.allclose(tg_inc.min_z[occ], tg_full.min_z[occ], atol=1e-5)
            assert np.allclose(tg_inc.max_z[occ], tg_full.max_z[occ], atol=1e-5)
            assert np.allclose(tg_inc.mean_z[occ], tg_full.mean_z[occ], atol=1e-5)
            assert np.allclose(tg_inc.mean_intensity[occ], tg_full.mean_intensity[occ], atol=1e-5)

    def test_incremental_world_query_equivalence(self, base_mapping_config):
        """Validates that world_query() on incremental map matches full rebuild queries."""
        gen = SyntheticLidarGenerator(seed=77)
        frame = gen.generate_frame(frame_id=0, num_ground_points=3000)

        _, _, dec = run_pipeline(frame, base_mapping_config)
        full_map = AdaptiveMap25DBuilder(config=base_mapping_config).build(frame, dec)

        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        inc_res = inc_builder.build(full_map, frame, dec)
        inc_map = inc_res.adaptive_map

        # Sample spatial test coordinates
        for x in np.linspace(-8, 8, 15):
            for y in np.linspace(-8, 8, 15):
                q_full = full_map.world_query(x, y)
                q_inc = inc_map.world_query(x, y)
                assert q_full.is_represented == q_inc.is_represented
                if q_full.is_represented:
                    assert q_full.tier == q_inc.tier
                    assert q_full.resolution == q_inc.resolution
                    assert abs(q_full.mean_z - q_inc.mean_z) < 1e-5


# ---------------------------------------------------------------------------
# 12. Temporal Map Manager Integration
# ---------------------------------------------------------------------------

class TestTemporalMapManagerIntegration:

    def test_manager_incremental_stream_workflow(self, base_mapping_config):
        """Verifies multi-frame stream processing through TemporalMapManager with incremental builder."""
        manager = TemporalMapManager(TemporalChangeConfig(elevation_change_threshold=0.20))
        inc_builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        gen = SyntheticLidarGenerator(seed=123)

        # Frame 0
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        _, _, dec0 = run_pipeline(f0, base_mapping_config)
        r0 = inc_builder.build(None, f0, dec0)
        manager.update(r0.adaptive_map)
        assert manager.frame_count == 1

        # Frame 1
        f1 = LidarFrame(points=f0.points.copy(), frame_id=1, timestamp=0.1)
        _, _, dec1 = run_pipeline(f1, base_mapping_config)
        r1 = inc_builder.build(manager.last_map, f1, dec1)
        manager.update(r1.adaptive_map)
        assert manager.frame_count == 2
        assert r1.reused_ratio == 1.0


# ---------------------------------------------------------------------------
# 13. Phase 13 Part C — Fast Path, Fallback, Memory Safety & Transitions Tests
# ---------------------------------------------------------------------------

class TestPhase13PartCOptimizations:

    def test_complete_reuse_fast_path(self, base_mapping_config):
        """Identical scene triggers complete_reuse_fast_path strategy and directly returns previous map."""
        gen = SyntheticLidarGenerator(seed=42)
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        _, _, dec0 = run_pipeline(f0, base_mapping_config)

        builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        r0 = builder.build(None, f0, dec0)

        # Same frame -> fast path
        r1 = builder.build(r0.adaptive_map, f0, dec0)
        assert r1.metadata["strategy"] == "FULL_REUSE"
        assert r1.metadata["status"] == "complete_reuse_fast_path"
        assert r1.adaptive_map is r0.adaptive_map  # Returned by reference directly

    def test_full_tier_reuse_by_reference(self, base_mapping_config):
        """An entire unchanged tier grid is reused by reference directly when spatial bounds are identical."""
        gen = SyntheticLidarGenerator(seed=42)
        f0 = gen.generate_frame(frame_id=0, num_ground_points=3000)
        _, _, dec0 = run_pipeline(f0, base_mapping_config)

        builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        r0 = builder.build(None, f0, dec0)

        # Modify only one point in a coarse cell (coarse tier changes, other tiers stable)
        pts1 = f0.points.copy()
        # Find a point in coarse tier
        is_coarse = (r0.adaptive_map.index_tier == "coarse")
        br, bc = np.where(is_coarse)
        if len(br) > 0:
            # Shift a point slightly in that cell to trigger coarse rebuild
            wx = base_mapping_config.min_x + (bc[0] + 0.5) * base_mapping_config.resolution
            wy = base_mapping_config.min_y + (br[0] + 0.5) * base_mapping_config.resolution
            dist = np.hypot(pts1[:, 0] - wx, pts1[:, 1] - wy)
            pts1[np.argmin(dist), 2] += 0.85

        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        _, _, dec1 = run_pipeline(f1, base_mapping_config)

        r1 = builder.build(r0.adaptive_map, f1, dec1)
        # Verify other tiers were reused by reference
        for t in ["medium", "fine", "ultra_fine"]:
            if t in r0.adaptive_map.tier_maps and t in r1.adaptive_map.tier_maps:
                # Same GridMap25D object reference
                assert r1.adaptive_map.tier_maps[t] is r0.adaptive_map.tier_maps[t]
                assert t in r1.reused_tiers

    def test_full_rebuild_fallback_triggered(self, base_mapping_config):
        """When rebuild ratio exceeds threshold, fallback to full builder is executed."""
        gen = SyntheticLidarGenerator(seed=42)
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        _, _, dec0 = run_pipeline(f0, base_mapping_config)

        # Low threshold (10%) to guarantee fallback triggers on minor changes
        builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config, full_rebuild_threshold=0.10)
        r0 = builder.build(None, f0, dec0)

        # Create massive changes (70% cells)
        pts1 = f0.points.copy()
        pts1[:, 2] += 0.95
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        _, _, dec1 = run_pipeline(f1, base_mapping_config)

        r1 = builder.build(r0.adaptive_map, f1, dec1)
        assert r1.metadata["strategy"] == "FULL_REBUILD"
        assert r1.metadata["status"] == "full_rebuild_fallback"

    def test_reused_maps_not_mutated(self, base_mapping_config):
        """Reusing previous maps does not mutate any of the arrays in the original grid map objects."""
        gen = SyntheticLidarGenerator(seed=12)
        f0 = gen.generate_frame(frame_id=0, num_ground_points=2000)
        _, _, dec0 = run_pipeline(f0, base_mapping_config)

        builder = IncrementalAdaptiveMapBuilder(config=base_mapping_config)
        r0 = builder.build(None, f0, dec0)

        # Snapshot all tier arrays before building r1
        orig_tiers_snapshot = {
            t: {
                "point_count": tg.point_count.copy(),
                "mean_z": tg.mean_z.copy(),
                "min_z": tg.min_z.copy(),
                "max_z": tg.max_z.copy(),
            }
            for t, tg in r0.adaptive_map.tier_maps.items()
        }

        # Build consecutive map with some elevation changes
        pts1 = f0.points.copy()
        pts1[:50, 2] += 1.50
        f1 = LidarFrame(points=pts1, frame_id=1, timestamp=0.1)
        _, _, dec1 = run_pipeline(f1, base_mapping_config)

        r1 = builder.build(r0.adaptive_map, f1, dec1)

        # Verify previous map tier arrays are untouched after building r1
        # Use allclose(equal_nan=True) because np.array_equal returns False for NaN==NaN
        for t, snap in orig_tiers_snapshot.items():
            if t not in r0.adaptive_map.tier_maps:
                continue
            tg = r0.adaptive_map.tier_maps[t]
            assert np.array_equal(tg.point_count, snap["point_count"]), f"{t} point_count mutated"
            assert np.allclose(tg.mean_z, snap["mean_z"], equal_nan=True), f"{t} mean_z mutated"
            assert np.allclose(tg.min_z, snap["min_z"], equal_nan=True), f"{t} min_z mutated"
            assert np.allclose(tg.max_z, snap["max_z"], equal_nan=True), f"{t} max_z mutated"

    def test_invalid_rebuild_threshold_raises_value_error(self, base_mapping_config):
        """Validates that a threshold outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="full_rebuild_threshold must be within"):
            IncrementalAdaptiveMapBuilder(config=base_mapping_config, full_rebuild_threshold=-0.1)
        with pytest.raises(ValueError, match="full_rebuild_threshold must be within"):
            IncrementalAdaptiveMapBuilder(config=base_mapping_config, full_rebuild_threshold=1.5)
