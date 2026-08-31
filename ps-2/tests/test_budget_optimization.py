"""
Unit and Integration Tests for Phase 10: Compute Budgeting and Resource-Aware Resolution Allocation.
"""

import pytest
import numpy as np
from pathlib import Path

from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.grid_map import GridMap25D
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.mapping.config import MappingConfig
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.priority.result import ResolutionDecisionResult
from backend.temporal.result import TemporalChangeResult
from backend.metrics.map_metrics import calculate_adaptive_map_metrics

from backend.budget.config import BudgetConfig
from backend.budget.cost_model import ResolutionCostModel, DOWNGRADE_ORDER, BYTES_PER_GRID_CELL
from backend.budget.estimator import ResourceEstimate, BudgetEstimator
from backend.budget.result import BudgetOptimizationResult
from backend.budget.optimizer import BudgetAwareResolutionOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_frame():
    gen = SyntheticLidarGenerator(seed=42)
    return gen.generate_frame(frame_id=0, num_ground_points=4000)


@pytest.fixture
def sample_decision_and_terrain(synthetic_frame):
    map_cfg = MappingConfig(resolution=0.5, min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0)
    prep_cfg = PreprocessingConfig()
    terrain_cfg = TerrainConfig()
    priority_cfg = PriorityConfig()

    prep_frame = PreprocessingPipeline(config=prep_cfg).process(synthetic_frame)
    base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(prep_frame)
    terrain_result = TerrainAnalyzer(config=terrain_cfg).analyze(base_grid)
    decision = AdaptiveResolutionSelector(config=priority_cfg).select_resolution(base_grid, terrain_result)

    return decision, terrain_result, prep_frame, map_cfg


# ---------------------------------------------------------------------------
# Test BudgetConfig
# ---------------------------------------------------------------------------

class TestBudgetConfig:

    def test_default_config(self):
        cfg = BudgetConfig()
        assert cfg.enabled is True
        assert cfg.max_memory_mb is not None
        assert cfg.max_allocated_cells is not None
        assert cfg.max_compute_cost is not None
        assert cfg.preserve_high_priority_regions is True
        assert cfg.allow_resolution_downgrade is True

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="max_memory_mb must be positive"):
            BudgetConfig(max_memory_mb=-1.0)

        with pytest.raises(ValueError, match="max_allocated_cells must be positive"):
            BudgetConfig(max_allocated_cells=0)

        with pytest.raises(ValueError, match="temporal_change_weight must be >= 1.0"):
            BudgetConfig(temporal_change_weight=0.5)

    def test_from_yaml(self):
        config_path = Path("configs/config.yaml")
        if config_path.is_file():
            cfg = BudgetConfig.from_yaml(config_path)
            assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Test ResolutionCostModel
# ---------------------------------------------------------------------------

class TestResolutionCostModel:

    def test_cost_increases_with_higher_resolution(self):
        model = ResolutionCostModel()
        base_res = 0.20

        cost_coarse = model.estimate_tier_compute_cost(100, 0.50, base_res)
        cost_medium = model.estimate_tier_compute_cost(100, 0.25, base_res)
        cost_fine = model.estimate_tier_compute_cost(100, 0.10, base_res)
        cost_ultra = model.estimate_tier_compute_cost(100, 0.05, base_res)

        assert cost_ultra > cost_fine > cost_medium > cost_coarse
        assert pytest.approx(cost_ultra, 0.01) == 1600.0  # 100 * (0.2/0.05)^2 = 1600
        assert pytest.approx(cost_coarse, 0.01) == 16.0   # 100 * (0.2/0.50)^2 = 16

    def test_downgrade_hierarchy_order(self):
        assert DOWNGRADE_ORDER["ultra_fine"] == "fine"
        assert DOWNGRADE_ORDER["fine"] == "medium"
        assert DOWNGRADE_ORDER["medium"] == "coarse"
        assert DOWNGRADE_ORDER["coarse"] is None

    def test_memory_estimation(self):
        model = ResolutionCostModel()
        cells = model.estimate_tier_cells(10, 0.10, 0.20)
        # 10 base cells @ 0.10m tier / 0.20m base = 10 * 4 = 40 fine cells
        assert cells == 40
        mem_bytes = model.estimate_tier_memory_bytes(10, 0.10, 0.20)
        assert mem_bytes == 40 * BYTES_PER_GRID_CELL


# ---------------------------------------------------------------------------
# Test BudgetEstimator
# ---------------------------------------------------------------------------

class TestBudgetEstimator:

    def test_estimate_decision(self, sample_decision_and_terrain):
        decision, _, _, _ = sample_decision_and_terrain
        estimator = BudgetEstimator()

        est = estimator.estimate_decision(decision)
        assert est.allocated_cells > 0
        assert est.memory_bytes > 0
        assert est.memory_mb > 0.0
        assert est.compute_cost > 0.0
        assert "ultra_fine" in est.tier_counts
        assert "coarse" in est.tier_counts

    def test_budget_fit_checking(self):
        estimator = BudgetEstimator()
        est = ResourceEstimate(
            allocated_cells=50000,
            memory_bytes=2 * 1024 * 1024,
            memory_mb=2.0,
            compute_cost=3000.0,
            tier_counts={"coarse": 1000},
            tier_allocated_cells={"coarse": 160},
            tier_memory_bytes={"coarse": 6400},
            tier_compute_cost={"coarse": 160.0},
        )

        # Generous
        cfg_ok = BudgetConfig(max_memory_mb=5.0, max_allocated_cells=100000, max_compute_cost=5000.0)
        fits, violations = estimator.check_budget_fit(est, cfg_ok)
        assert fits is True
        assert len(violations) == 0

        # Memory exceeded
        cfg_mem = BudgetConfig(max_memory_mb=1.0)
        fits, violations = estimator.check_budget_fit(est, cfg_mem)
        assert fits is False
        assert any("Memory limit exceeded" in v for v in violations)

        # Compute cost exceeded
        cfg_cost = BudgetConfig(max_compute_cost=2000.0)
        fits, violations = estimator.check_budget_fit(est, cfg_cost)
        assert fits is False
        assert any("Compute cost limit exceeded" in v for v in violations)


# ---------------------------------------------------------------------------
# Test BudgetAwareResolutionOptimizer
# ---------------------------------------------------------------------------

class TestBudgetAwareResolutionOptimizer:

    def test_generous_budget_preserves_original(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        optimizer = BudgetAwareResolutionOptimizer(
            config=BudgetConfig(max_memory_mb=50.0, max_allocated_cells=1000000, max_compute_cost=100000.0)
        )

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)

        assert result.status == "within_budget"
        assert result.is_downgraded is False
        assert result.num_downgraded_cells == 0
        assert result.original_fits_budget is True
        assert result.optimized_fits_budget is True
        assert result.optimized_counts == result.original_counts

    def test_disabled_budget_leaves_unchanged(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        optimizer = BudgetAwareResolutionOptimizer(
            config=BudgetConfig(enabled=False, max_compute_cost=10.0)
        )

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)
        assert result.status == "disabled"
        assert result.is_downgraded is False

    def test_moderate_budget_causes_valid_downgrades(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        estimator = BudgetEstimator()
        orig_est = estimator.estimate_decision(decision)

        # Set budget to 60% of original compute cost
        target_cost = orig_est.compute_cost * 0.60
        cfg = BudgetConfig(max_compute_cost=target_cost, max_memory_mb=50.0, max_allocated_cells=500000)
        optimizer = BudgetAwareResolutionOptimizer(config=cfg)

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)

        assert result.status == "optimized_success"
        assert result.is_downgraded is True
        assert result.num_downgraded_cells > 0
        assert result.optimized_estimate.compute_cost <= target_cost
        assert result.optimized_fits_budget is True

    def test_optimizer_never_upgrades_resolution(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        cfg = BudgetConfig(max_compute_cost=500.0, max_memory_mb=50.0, max_allocated_cells=500000)
        optimizer = BudgetAwareResolutionOptimizer(config=cfg)

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)

        # For every cell, optimized resolution must be >= original resolution (numerically larger = coarser)
        orig_res = decision.assigned_resolution[decision.valid_mask]
        opt_res = result.optimized_decision.assigned_resolution[decision.valid_mask]

        # Resolution numerical value: 0.05 (fine) <= 0.10 <= 0.25 <= 0.50 (coarse)
        assert np.all(opt_res >= orig_res - 1e-5)

    def test_priority_preservation_order(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        estimator = BudgetEstimator()
        orig_est = estimator.estimate_decision(decision)

        # Force a downgrade of some ultra_fine cells
        cfg = BudgetConfig(max_compute_cost=orig_est.compute_cost * 0.8, max_memory_mb=50.0, max_allocated_cells=500000)
        optimizer = BudgetAwareResolutionOptimizer(config=cfg)

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)

        # Cells that remained ultra_fine must have higher average roughness than cells downgraded to fine
        still_ultra = (decision.assigned_level == "ultra_fine") & (result.optimized_decision.assigned_level == "ultra_fine")
        downgraded_from_ultra = (decision.assigned_level == "ultra_fine") & (result.optimized_decision.assigned_level != "ultra_fine")

        if np.any(still_ultra) and np.any(downgraded_from_ultra):
            rough_still = np.nanmean(terrain_result.roughness[still_ultra])
            rough_down = np.nanmean(terrain_result.roughness[downgraded_from_ultra])
            assert rough_still >= rough_down

    def test_impossible_minimum_budget(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        # Impossibly low limits
        cfg = BudgetConfig(max_memory_mb=0.01, max_allocated_cells=10, max_compute_cost=1.0)
        optimizer = BudgetAwareResolutionOptimizer(config=cfg)

        result = optimizer.optimize(decision=decision, terrain_result=terrain_result)

        assert result.status == "impossible_budget"
        assert result.optimized_fits_budget is False
        assert len(result.reasons) > 0
        assert "exceeds budget" in result.reasons[0]

    def test_temporal_change_priority_boost(self, sample_decision_and_terrain):
        decision, terrain_result, _, _ = sample_decision_and_terrain
        rows, cols = decision.grid_map.shape

        # Create a mock temporal result where 5 ultra_fine cells are dynamically changed
        changed_mask = np.zeros((rows, cols), dtype=bool)
        ultra_indices = np.argwhere(decision.valid_mask & (decision.assigned_level == "ultra_fine"))
        if len(ultra_indices) > 5:
            for r, c in ultra_indices[:5]:
                changed_mask[r, c] = True

        mock_temporal = TemporalChangeResult(
            frame_index=1,
            previous_frame_id=0,
            current_frame_id=1,
            grid_shape=(rows, cols),
            resolution=decision.grid_map.resolution,
            bounds=(decision.grid_map.min_x, decision.grid_map.max_x, decision.grid_map.min_y, decision.grid_map.max_y),
            is_initial_frame=False,
            new_occupancy_mask=changed_mask,
            removed_occupancy_mask=np.zeros((rows, cols), dtype=bool),
            elevation_changed_mask=np.zeros((rows, cols), dtype=bool),
            elevation_difference=np.full((rows, cols), np.nan, dtype=np.float32),
            changed_mask=changed_mask,
            stable_mask=~changed_mask,
            unoccupied_mask=np.zeros((rows, cols), dtype=bool),
        )

        estimator = BudgetEstimator()
        orig_est = estimator.estimate_decision(decision)
        cfg = BudgetConfig(max_compute_cost=orig_est.compute_cost * 0.7, temporal_change_weight=2.0)
        optimizer = BudgetAwareResolutionOptimizer(config=cfg)

        result = optimizer.optimize(
            decision=decision,
            terrain_result=terrain_result,
            temporal_result=mock_temporal,
        )

        # The 5 changed cells should be preserved in ultra_fine
        for r, c in ultra_indices[:5]:
            assert result.optimized_decision.assigned_level[r, c] == "ultra_fine"


# ---------------------------------------------------------------------------
# Test End-to-End Pipeline Integration with Budgeting
# ---------------------------------------------------------------------------

class TestBudgetPipelineIntegration:

    def test_full_pipeline_with_budget_optimizer(self, synthetic_frame):
        # 1. Pipeline stages
        prep = PreprocessingPipeline(config=PreprocessingConfig()).process(synthetic_frame)
        map_cfg = MappingConfig(resolution=0.5, min_x=-15.0, max_x=15.0, min_y=-15.0, max_y=15.0)
        base_grid = Uniform25DMapBuilder(config=map_cfg).build_map(prep)
        terrain = TerrainAnalyzer(config=TerrainConfig()).analyze(base_grid)
        decision = AdaptiveResolutionSelector(config=PriorityConfig()).select_resolution(base_grid, terrain)

        # 2. Budget optimization
        budget_cfg = BudgetConfig(max_compute_cost=3000.0, max_memory_mb=2.0)
        optimizer = BudgetAwareResolutionOptimizer(config=budget_cfg)
        budget_result = optimizer.optimize(decision=decision, terrain_result=terrain)

        assert isinstance(budget_result, BudgetOptimizationResult)

        # 3. Build final AdaptiveMap25D from budget-constrained decisions
        adaptive_builder = AdaptiveMap25DBuilder(config=map_cfg)
        adaptive_map = adaptive_builder.build(prep, budget_result.optimized_decision)

        assert adaptive_map.num_represented_cells > 0
        metrics = calculate_adaptive_map_metrics(adaptive_map)
        assert metrics.memory.array_bytes > 0
