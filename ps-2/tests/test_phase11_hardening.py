"""
Phase 11 Hardening & Regression Test Suite.

Tests for:
  - Vectorized terrain analyzer numerical equivalence and correctness
  - Boundary conditions (inclusive limits, corner points, max bounds)
  - Sparse neighbor thresholds and NaN/empty grid handling
  - Vectorized resolution selector decision equivalence
  - Memory / cost model physical accuracy (20 bytes/cell)
  - Benchmark metric clarity and separation
  - End-to-end run_demo pipeline execution
"""

import numpy as np
import pytest
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator, SyntheticLidarLoader
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.grid_map import GridMap25D
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.terrain.result import TerrainAnalysisResult
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.priority.result import ResolutionDecisionResult
from backend.budget.cost_model import ResolutionCostModel, ACTUAL_BYTES_PER_GRID_CELL, BYTES_PER_GRID_CELL
from backend.budget.config import BudgetConfig
from backend.budget.estimator import BudgetEstimator, ResourceEstimate
from backend.budget.optimizer import BudgetAwareResolutionOptimizer
from backend.metrics.map_metrics import calculate_grid_map_metrics, calculate_adaptive_map_metrics

import run_demo


# ===========================================================================
# 1. Terrain Analyzer Numerical Equivalence & Correctness
# ===========================================================================

class TestTerrainAnalyzerHardening:

    def test_vectorized_terrain_numerical_equivalence(self):
        """Validates that vectorized TerrainAnalyzer matches reference cell-by-cell calculations."""
        rows, cols = 40, 40
        res = 0.5
        rng = np.random.default_rng(123)
        mean_z = rng.uniform(-2.0, 5.0, size=(rows, cols)).astype(np.float32)

        # Introduce realistic sparsity (30% occupied)
        occupied_mask = rng.uniform(0, 1, size=(rows, cols)) < 0.30
        mean_z[~occupied_mask] = np.nan

        point_count = np.where(occupied_mask, rng.integers(1, 10, size=(rows, cols)), 0).astype(np.int32)
        min_z = np.where(occupied_mask, mean_z - rng.uniform(0.1, 0.5, size=(rows, cols)), np.nan).astype(np.float32)
        max_z = np.where(occupied_mask, mean_z + rng.uniform(0.1, 0.5, size=(rows, cols)), np.nan).astype(np.float32)
        mean_intensity = np.where(occupied_mask, rng.uniform(0.2, 0.8, size=(rows, cols)), np.nan).astype(np.float32)

        grid = GridMap25D(
            min_x=-10.0, max_x=10.0,
            min_y=-10.0, max_y=10.0,
            resolution=res,
            point_count=point_count,
            min_z=min_z,
            max_z=max_z,
            mean_z=mean_z,
            mean_intensity=mean_intensity,
        )

        cfg = TerrainConfig(window_size=3, min_valid_neighbors=1)
        analyzer = TerrainAnalyzer(config=cfg)
        result = analyzer.analyze(grid)

        # Independent reference loop implementation
        w = cfg.window_size
        half_w = w // 2
        ref_roughness = np.full((rows, cols), np.nan, dtype=np.float32)
        ref_range = np.full((rows, cols), np.nan, dtype=np.float32)
        ref_slope = np.full((rows, cols), np.nan, dtype=np.float32)
        ref_mask = np.zeros((rows, cols), dtype=bool)

        for r in range(rows):
            for c in range(cols):
                if not occupied_mask[r, c]:
                    continue
                r_min = max(0, r - half_w)
                r_max = min(rows, r + half_w + 1)
                c_min = max(0, c - half_w)
                c_max = min(cols, c + half_w + 1)
                win = mean_z[r_min:r_max, c_min:c_max]
                valid_vals = win[~np.isnan(win)]
                if len(valid_vals) < cfg.min_valid_neighbors:
                    continue

                ref_roughness[r, c] = float(np.std(valid_vals))
                ref_range[r, c] = float(np.max(valid_vals) - np.min(valid_vals))

                center = mean_z[r, c]
                has_l = (c > 0) and not np.isnan(mean_z[r, c - 1])
                has_r = (c < cols - 1) and not np.isnan(mean_z[r, c + 1])
                if has_l and has_r:
                    dz_dx = (mean_z[r, c + 1] - mean_z[r, c - 1]) / (2.0 * res)
                elif has_r:
                    dz_dx = (mean_z[r, c + 1] - center) / res
                elif has_l:
                    dz_dx = (center - mean_z[r, c - 1]) / res
                else:
                    dz_dx = 0.0

                has_u = (r > 0) and not np.isnan(mean_z[r - 1, c])
                has_d = (r < rows - 1) and not np.isnan(mean_z[r + 1, c])
                if has_u and has_d:
                    dz_dy = (mean_z[r + 1, c] - mean_z[r - 1, c]) / (2.0 * res)
                elif has_d:
                    dz_dy = (mean_z[r + 1, c] - center) / res
                elif has_u:
                    dz_dy = (center - mean_z[r - 1, c]) / res
                else:
                    dz_dy = 0.0

                ref_slope[r, c] = float(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
                ref_mask[r, c] = True

        assert np.array_equal(result.analyzed_mask, ref_mask)
        valid = result.analyzed_mask
        assert np.allclose(result.roughness[valid], ref_roughness[valid], atol=1e-5)
        assert np.allclose(result.elevation_range[valid], ref_range[valid], atol=1e-5)
        assert np.allclose(result.slope[valid], ref_slope[valid], atol=1e-5)

    def test_min_valid_neighbors_threshold(self):
        """Verifies that cells with fewer valid neighbors than threshold are skipped."""
        rows, cols = 10, 10
        mean_z = np.full((rows, cols), np.nan, dtype=np.float32)
        # Single isolated occupied cell at (5, 5)
        mean_z[5, 5] = 1.0
        point_count = np.where(~np.isnan(mean_z), 1, 0).astype(np.int32)

        grid = GridMap25D(
            min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, resolution=1.0,
            point_count=point_count, min_z=mean_z.copy(), max_z=mean_z.copy(),
            mean_z=mean_z.copy(), mean_intensity=np.zeros_like(mean_z),
        )

        # With min_valid_neighbors=2, the isolated cell (which has only 1 valid value) should be skipped
        cfg = TerrainConfig(window_size=3, min_valid_neighbors=2)
        result = TerrainAnalyzer(config=cfg).analyze(grid)

        assert result.analyzed_mask[5, 5] is np.False_ or result.analyzed_mask[5, 5] == False
        assert np.isnan(result.roughness[5, 5])
        assert result.num_analyzed_cells == 0

    def test_empty_grid_analysis(self):
        """Verifies completely empty grid map returns all NaN feature layers."""
        grid = Uniform25DMapBuilder(
            MappingConfig(resolution=1.0, min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        result = TerrainAnalyzer().analyze(grid)
        assert result.num_analyzed_cells == 0
        assert np.all(np.isnan(result.roughness))
        assert np.all(np.isnan(result.elevation_range))
        assert np.all(np.isnan(result.slope))


# ===========================================================================
# 2. Vectorized Resolution Selection Equivalence
# ===========================================================================

class TestResolutionSelectorHardening:

    def test_vectorized_selector_equivalence(self):
        """Verifies vectorized AdaptiveResolutionSelector matches threshold decision rules."""
        rows, cols = 30, 30
        rng = np.random.default_rng(42)
        roughness = rng.uniform(0.0, 0.40, size=(rows, cols)).astype(np.float32)
        slope = rng.uniform(0.0, 2.0, size=(rows, cols)).astype(np.float32)
        elevation_range = rng.uniform(0.0, 1.2, size=(rows, cols)).astype(np.float32)

        analyzed_mask = rng.uniform(0, 1, size=(rows, cols)) < 0.50
        roughness[~analyzed_mask] = np.nan
        slope[~analyzed_mask] = np.nan
        elevation_range[~analyzed_mask] = np.nan

        grid = GridMap25D(
            min_x=0.0, max_x=30.0, min_y=0.0, max_y=30.0, resolution=1.0,
            point_count=np.where(analyzed_mask, 5, 0).astype(np.int32),
            min_z=np.zeros((rows, cols), dtype=np.float32),
            max_z=np.ones((rows, cols), dtype=np.float32),
            mean_z=np.full((rows, cols), 0.5, dtype=np.float32),
            mean_intensity=np.full((rows, cols), 0.5, dtype=np.float32),
        )

        terrain_res = TerrainAnalysisResult(
            grid_map=grid,
            roughness=roughness,
            elevation_range=elevation_range,
            slope=slope,
            analyzed_mask=analyzed_mask,
        )

        cfg = PriorityConfig()
        selector = AdaptiveResolutionSelector(config=cfg)
        result = selector.select_resolution(grid, terrain_res)

        # Verify tier counts sum to total decided
        total_decided = result.num_decided_cells
        assert sum(result.counts.values()) == total_decided

        # Verify cell-by-cell rule satisfaction
        thresholds = cfg.thresholds
        t_uf = thresholds["ultra_fine"]
        t_fine = thresholds["fine"]
        t_med = thresholds["medium"]

        for r in range(rows):
            for c in range(cols):
                if not analyzed_mask[r, c]:
                    assert result.assigned_level[r, c] == ""
                    assert np.isnan(result.assigned_resolution[r, c])
                    continue

                r_val = roughness[r, c]
                s_val = slope[r, c]
                e_val = elevation_range[r, c]
                level = result.assigned_level[r, c]

                if r_val >= t_uf["roughness"] or s_val >= t_uf["slope"] or e_val >= t_uf["elevation_range"]:
                    assert level == "ultra_fine"
                elif r_val >= t_fine["roughness"] or s_val >= t_fine["slope"] or e_val >= t_fine["elevation_range"]:
                    assert level == "fine"
                elif r_val >= t_med["roughness"] or s_val >= t_med["slope"] or e_val >= t_med["elevation_range"]:
                    assert level == "medium"
                else:
                    assert level == "coarse"


# ===========================================================================
# 3. Boundary Consistency & Max Coordinate Clamping
# ===========================================================================

class TestBoundaryConsistency:

    def test_corner_points_ingestion(self):
        """Verifies points placed exactly at all 4 boundary corners are retained and mapped."""
        min_x, max_x = -10.0, 10.0
        min_y, max_y = -10.0, 10.0
        res = 1.0

        pts = np.array([
            [min_x, min_y, 1.0, 0.5],  # Bottom-Left
            [min_x, max_y, 2.0, 0.5],  # Top-Left
            [max_x, min_y, 3.0, 0.5],  # Bottom-Right
            [max_x, max_y, 4.0, 0.5],  # Top-Right
        ], dtype=np.float32)

        frame = LidarFrame(points=pts)
        builder = Uniform25DMapBuilder(
            MappingConfig(resolution=res, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        )
        grid = builder.build_map(frame)

        # All 4 points should be mapped
        assert grid.metadata["mapped_points"] == 4
        assert grid.point_count[0, 0] >= 1       # (min_x, min_y) -> (0, 0)
        assert grid.point_count[grid.rows - 1, 0] >= 1   # (min_x, max_y) -> (rows-1, 0)
        assert grid.point_count[0, grid.cols - 1] >= 1   # (max_x, min_y) -> (0, cols-1)
        assert grid.point_count[grid.rows - 1, grid.cols - 1] >= 1  # (max_x, max_y) -> (rows-1, cols-1)

    def test_adaptive_builder_boundary_point_retention(self):
        """Verifies AdaptiveMap25DBuilder retains boundary points matching uniform builder."""
        min_x, max_x = -10.0, 10.0
        min_y, max_y = -10.0, 10.0
        res = 1.0

        pts = np.array([
            [max_x, max_y, 3.5, 0.8],
            [min_x, min_y, 1.2, 0.3],
            [0.0, 0.0, 0.5, 0.5],
        ], dtype=np.float32)

        frame = LidarFrame(points=pts)
        map_cfg = MappingConfig(resolution=res, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(frame)
        terrain_res = TerrainAnalyzer().analyze(base_grid)
        decision = AdaptiveResolutionSelector().select_resolution(base_grid, terrain_res)

        adaptive_map = AdaptiveMap25DBuilder(config=map_cfg).build(frame, decision)

        # Query points on boundaries
        q_max = adaptive_map.world_query(max_x, max_y)
        q_min = adaptive_map.world_query(min_x, min_y)

        assert adaptive_map.in_bounds(max_x, max_y) is True
        assert adaptive_map.in_bounds(min_x, min_y) is True
        assert q_max.is_represented is True
        assert q_min.is_represented is True


# ===========================================================================
# 4. Memory / Cost Model Physical Accuracy (20 Bytes/Cell)
# ===========================================================================

class TestMemoryModelAccuracy:

    def test_grid_map_physical_bytes_per_cell(self):
        """Verifies that GridMap25D physical array storage is exactly 20 bytes/cell."""
        rows, cols = 100, 100
        grid = Uniform25DMapBuilder(
            MappingConfig(resolution=0.2, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        ).build_map(LidarFrame(points=np.empty((0, 4), dtype=np.float32)))

        metrics = calculate_grid_map_metrics(grid)
        total_cells = grid.total_cells  # 10,000

        # Layers: point_count (int32 = 4B), min_z/max_z/mean_z/mean_intensity (4 * float32 = 16B) = 20B
        expected_bytes = total_cells * 20
        assert metrics.memory.array_bytes == expected_bytes
        assert ACTUAL_BYTES_PER_GRID_CELL == 20
        assert BYTES_PER_GRID_CELL == 20

    def test_cost_model_deterministic_calculation(self):
        """Verifies ResolutionCostModel produces deterministic estimates using 20 bytes/cell."""
        model = ResolutionCostModel()
        num_base_cells = 50
        tier_res = 0.10
        base_res = 0.20

        # Subdivisions: (0.20 / 0.10)^2 = 4 fine cells per base cell -> 200 fine cells
        fine_cells = model.estimate_tier_cells(num_base_cells, tier_res, base_res)
        assert fine_cells == 200

        # Memory: 200 cells * 20 bytes/cell = 4000 bytes
        mem_bytes = model.estimate_tier_memory_bytes(num_base_cells, tier_res, base_res)
        assert mem_bytes == 4000


# ===========================================================================
# 5. Benchmark Metric Clarity & Separation
# ===========================================================================

class TestBenchmarkMetricClarity:

    def test_adaptive_map_metric_separation(self):
        """Verifies that AdaptiveMap metrics distinguish tier-allocated cells vs base index cells."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(num_ground_points=2000)

        map_cfg = MappingConfig(resolution=0.5, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        clean = PreprocessingPipeline().process(frame)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(clean)
        terr = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terr)
        amap = AdaptiveMap25DBuilder(config=map_cfg).build(clean, dec)

        metrics = calculate_adaptive_map_metrics(amap)
        assert metrics.adaptive_breakdown is not None
        bd = metrics.adaptive_breakdown

        assert bd.base_decision_cells == amap.base_shape[0] * amap.base_shape[1]
        assert bd.tier_allocated_cells > 0
        assert metrics.size.total_cells == bd.tier_allocated_cells + bd.base_decision_cells
        assert metrics.memory.array_bytes > 0


# ===========================================================================
# 6. Vectorized Adaptive Index Matrix Verification (Phase 12 Part B)
# ===========================================================================

class TestVectorizedAdaptiveIndexConstruction:

    def test_vectorized_index_matrices_correctness(self):
        """Verifies index_tier, index_row, index_col match exact cell center mapping."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(num_ground_points=4000)

        map_cfg = MappingConfig(resolution=0.5, min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0)
        clean = PreprocessingPipeline().process(frame)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(clean)
        terr = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terr)
        amap = AdaptiveMap25DBuilder(config=map_cfg).build(clean, dec)

        rows, cols = amap.base_shape

        # Unassigned cells must have empty tier and -1 row/col
        unassigned_mask = ~dec.valid_mask
        assert np.all(amap.index_tier[unassigned_mask] == "")
        assert np.all(amap.index_row[unassigned_mask] == -1)
        assert np.all(amap.index_col[unassigned_mask] == -1)

        # Assigned cells must point to valid tier-grid cells
        for tier in amap.available_tiers:
            t_mask = dec.valid_mask & (dec.assigned_level == tier)
            assert np.all(amap.index_tier[t_mask] == tier)
            tier_grid = amap.tier_maps[tier]

            br, bc = np.where(t_mask)
            fr = amap.index_row[br, bc]
            fc = amap.index_col[br, bc]

            assert np.all(fr >= 0)
            assert np.all(fr < tier_grid.rows)
            assert np.all(fc >= 0)
            assert np.all(fc < tier_grid.cols)

            # Check world coordinate mapping consistency
            wx = base.min_x + (bc + 0.5) * base.resolution
            wy = base.min_y + (br + 0.5) * base.resolution
            expected_fc = np.clip(np.floor((wx - tier_grid.min_x) / tier_grid.resolution).astype(np.int32), 0, tier_grid.cols - 1)
            expected_fr = np.clip(np.floor((wy - tier_grid.min_y) / tier_grid.resolution).astype(np.int32), 0, tier_grid.rows - 1)

            assert np.array_equal(fr, expected_fr)
            assert np.array_equal(fc, expected_fc)

    def test_subset_of_tiers_budget_constrained(self):
        """Verifies adaptive index construction when only a subset of tiers is present."""
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(num_ground_points=4000)

        map_cfg = MappingConfig(resolution=0.5, min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0)
        clean = PreprocessingPipeline().process(frame)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(clean)
        terr = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terr)

        # Apply tight budget to force downgrades (only coarse and medium active)
        budget_cfg = BudgetConfig(max_compute_cost=1200.0, max_memory_mb=1.5, max_allocated_cells=64000)
        opt_res = BudgetAwareResolutionOptimizer(config=budget_cfg).optimize(dec, terrain_result=terr)

        amap = AdaptiveMap25DBuilder(config=map_cfg).build(clean, opt_res.optimized_decision)
        assert len(amap.tier_maps) <= 2
        assert "coarse" in amap.tier_maps
        assert amap.num_represented_cells > 0


# ===========================================================================
# 7. End-to-End run_demo.py Execution
# ===========================================================================

class TestRunDemoExecution:

    def test_run_demo_executes_successfully(self, capsys):
        """Verifies run_demo.py executes the entire pipeline cleanly."""
        run_demo.run()
        captured = capsys.readouterr()
        assert "ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR MAPPING DEMO" in captured.out
        assert "PIPELINE PERFORMANCE:" in captured.out
        assert "End-to-end demonstration completed successfully." in captured.out


# ===========================================================================
# 8. Phase 12 Part C — Optimized Builder Correctness Regression Tests
# ===========================================================================

def _build_reference_tier_grid(prep, bg, tier, tier_res, valid_mask, assigned_level):
    """Reference: old _filter_frame + Uniform25DMapBuilder path (used for equivalence checks)."""
    from backend.data.frame import LidarFrame as LF
    tier_mask = valid_mask & (assigned_level == tier)
    tmx, tMx, tmy, tMy = AdaptiveMap25DBuilder._tier_world_bounds(bg, tier_mask, tier_res)
    tc = MappingConfig(resolution=tier_res, min_x=tmx, max_x=tMx, min_y=tmy, max_y=tMy, auto_bounds=False)
    pts = prep.points
    mask = (pts[:, 0] >= tmx) & (pts[:, 0] <= tMx) & (pts[:, 1] >= tmy) & (pts[:, 1] <= tMy)
    lf = LF(points=pts[mask].copy(), frame_id=prep.frame_id, timestamp=prep.timestamp, metadata=dict(prep.metadata))
    return Uniform25DMapBuilder(config=tc).build_map(lf)


class TestPhase12PartCCorrectness:

    @pytest.fixture
    def pipeline_output(self):
        gen = SyntheticLidarGenerator(seed=99)
        frame = gen.generate_frame(num_ground_points=5000)
        map_cfg = MappingConfig(resolution=0.5, min_x=-20.0, max_x=20.0, min_y=-20.0, max_y=20.0)
        prep = PreprocessingPipeline().process(frame)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(prep)
        terrain = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terrain)
        amap = AdaptiveMap25DBuilder(config=map_cfg).build(prep, dec)
        return prep, dec, amap, map_cfg

    def test_tier_maps_point_count_identical_to_reference(self, pipeline_output):
        """Tier point_count arrays must exactly match the old _filter_frame+build_map() path."""
        prep, dec, amap, map_cfg = pipeline_output
        bg = dec.grid_map
        for tier, tier_res in dec.resolution_levels.items():
            if tier not in amap.tier_maps:
                continue
            ref = _build_reference_tier_grid(prep, bg, tier, tier_res, dec.valid_mask, dec.assigned_level)
            got = amap.tier_maps[tier]
            assert np.array_equal(ref.point_count, got.point_count), \
                f"point_count mismatch in tier '{tier}'"

    def test_tier_maps_min_z_identical(self, pipeline_output):
        """Tier min_z (occupied cells) must be identical to reference within float32 tolerance."""
        prep, dec, amap, _ = pipeline_output
        bg = dec.grid_map
        for tier, tier_res in dec.resolution_levels.items():
            if tier not in amap.tier_maps:
                continue
            ref = _build_reference_tier_grid(prep, bg, tier, tier_res, dec.valid_mask, dec.assigned_level)
            got = amap.tier_maps[tier]
            occ = ~np.isnan(ref.min_z)
            assert np.array_equal(np.isnan(ref.min_z), np.isnan(got.min_z)), \
                f"min_z NaN pattern mismatch in tier '{tier}'"
            assert np.allclose(ref.min_z[occ], got.min_z[occ], atol=1e-5), \
                f"min_z value mismatch in tier '{tier}'"

    def test_tier_maps_max_z_identical(self, pipeline_output):
        """Tier max_z must be identical to reference."""
        prep, dec, amap, _ = pipeline_output
        bg = dec.grid_map
        for tier, tier_res in dec.resolution_levels.items():
            if tier not in amap.tier_maps:
                continue
            ref = _build_reference_tier_grid(prep, bg, tier, tier_res, dec.valid_mask, dec.assigned_level)
            got = amap.tier_maps[tier]
            occ = ~np.isnan(ref.max_z)
            assert np.allclose(ref.max_z[occ], got.max_z[occ], atol=1e-5), \
                f"max_z value mismatch in tier '{tier}'"

    def test_tier_maps_mean_z_identical(self, pipeline_output):
        """Tier mean_z must be identical to reference within float32 tolerance."""
        prep, dec, amap, _ = pipeline_output
        bg = dec.grid_map
        for tier, tier_res in dec.resolution_levels.items():
            if tier not in amap.tier_maps:
                continue
            ref = _build_reference_tier_grid(prep, bg, tier, tier_res, dec.valid_mask, dec.assigned_level)
            got = amap.tier_maps[tier]
            occ = ~np.isnan(ref.mean_z)
            assert np.allclose(ref.mean_z[occ], got.mean_z[occ], atol=1e-5), \
                f"mean_z value mismatch in tier '{tier}'"

    def test_tier_maps_mean_intensity_identical(self, pipeline_output):
        """Tier mean_intensity must be identical to reference within float32 tolerance."""
        prep, dec, amap, _ = pipeline_output
        bg = dec.grid_map
        for tier, tier_res in dec.resolution_levels.items():
            if tier not in amap.tier_maps:
                continue
            ref = _build_reference_tier_grid(prep, bg, tier, tier_res, dec.valid_mask, dec.assigned_level)
            got = amap.tier_maps[tier]
            occ = ~np.isnan(ref.mean_intensity)
            assert np.allclose(ref.mean_intensity[occ], got.mean_intensity[occ], atol=1e-5), \
                f"mean_intensity mismatch in tier '{tier}'"

    def test_boundary_points_retained_all_four_corners(self):
        """Points exactly on all four boundary corners must appear in the resulting map."""
        corners = np.array([
            [-10.0, -10.0, 1.0, 100.0],
            [ 10.0, -10.0, 2.0, 110.0],
            [-10.0,  10.0, 3.0, 120.0],
            [ 10.0,  10.0, 4.0, 130.0],
            # Interior background points
            *[[float(x), float(y), 0.5, 80.0]
              for x in np.linspace(-9, 9, 12) for y in np.linspace(-9, 9, 12)],
        ], dtype=np.float64)
        lf = LidarFrame(points=corners, frame_id=0, timestamp=0.0, metadata={})
        map_cfg = MappingConfig(resolution=0.5, min_x=-10.0, max_x=10.0, min_y=-10.0, max_y=10.0)
        prep = PreprocessingPipeline().process(lf)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(prep)
        terrain = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terrain)
        amap = AdaptiveMap25DBuilder(config=map_cfg).build(prep, dec)

        # Total represented points across all tiers >= number of distinct interior background pts
        total_pts = sum(int(g.point_count.sum()) for g in amap.tier_maps.values())
        assert total_pts >= len(corners), \
            f"Expected >= {len(corners)} total points, got {total_pts}"

    def test_unassigned_cells_have_sentinel_values(self, pipeline_output):
        """Cells not in decision.valid_mask must retain sentinel index values."""
        prep, dec, amap, _ = pipeline_output
        unassigned = ~dec.valid_mask
        assert np.all(amap.index_tier[unassigned] == ""), "unassigned index_tier must be ''"
        assert np.all(amap.index_row[unassigned] == -1), "unassigned index_row must be -1"
        assert np.all(amap.index_col[unassigned] == -1), "unassigned index_col must be -1"

    def test_subset_tiers_budget_constrained(self):
        """Build must work correctly when budget optimizer restricts to a subset of tiers."""
        gen = SyntheticLidarGenerator(seed=77)
        frame = gen.generate_frame(num_ground_points=4000)
        map_cfg = MappingConfig(resolution=0.5, min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0)
        prep = PreprocessingPipeline().process(frame)
        base = Uniform25DMapBuilder(config=map_cfg).build_map(prep)
        terrain = TerrainAnalyzer().analyze(base)
        dec = AdaptiveResolutionSelector().select_resolution(base, terrain)
        budget_cfg = BudgetConfig(max_compute_cost=1200.0, max_memory_mb=1.5, max_allocated_cells=64000)
        opt = BudgetAwareResolutionOptimizer(config=budget_cfg).optimize(dec, terrain_result=terrain)
        amap = AdaptiveMap25DBuilder(config=map_cfg).build(prep, opt.optimized_decision)
        assert len(amap.tier_maps) <= 2
        assert "coarse" in amap.tier_maps
        assert amap.num_represented_cells > 0

    def test_world_query_returns_identical_results(self, pipeline_output):
        """world_query() must return consistent data via both index lookup and direct tier access."""
        prep, dec, amap, _ = pipeline_output
        rows, cols = amap.base_shape
        bg = dec.grid_map

        queried = 0
        for r in range(0, rows, max(1, rows // 15)):
            for c in range(0, cols, max(1, cols // 15)):
                tier = amap.index_tier[r, c]
                if tier == "":
                    continue
                wx = bg.min_x + (c + 0.5) * bg.resolution
                wy = bg.min_y + (r + 0.5) * bg.resolution
                q = amap.world_query(wx, wy)
                if not q.is_represented:
                    continue  # Unoccupied cell (0 points)
                assert q.tier == tier
                assert q.resolution == amap.tier_resolutions[tier]
                fr = amap.index_row[r, c]
                fc = amap.index_col[r, c]
                tg = amap.tier_maps[tier]
                assert abs(q.mean_z - tg.mean_z[fr, fc]) < 1e-5
                queried += 1

        assert queried > 0, "No occupied cells were queried — check setup"

    def test_adaptive_map_build_signature_unchanged(self):
        """build() must accept (frame, decision) and return AdaptiveMap25D — API unchanged."""
        import inspect
        sig = inspect.signature(AdaptiveMap25DBuilder.build)
        params = list(sig.parameters.keys())
        assert params == ["self", "frame", "decision"]
        ann = sig.return_annotation
        # Accept both the resolved class and its forward-reference string form
        assert ann is AdaptiveMap25D or ann == "AdaptiveMap25D" or ann is inspect.Parameter.empty

