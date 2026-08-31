"""
Budget-Aware Resolution Optimizer.

Applies priority-preserving, resource-constrained resolution downgrades to ensure
spatial mapping plans satisfy memory, cell count, and compute budgets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

from backend.priority.result import ResolutionDecisionResult
from backend.terrain.result import TerrainAnalysisResult
from backend.temporal.result import TemporalChangeResult
from backend.budget.config import BudgetConfig
from backend.budget.cost_model import ResolutionCostModel
from backend.budget.estimator import BudgetEstimator, ResourceEstimate
from backend.budget.result import BudgetOptimizationResult


class BudgetAwareResolutionOptimizer:
    """
    Optimizes resolution decisions against resource constraints while preserving
    high-complexity and dynamically active terrain features.

    Priority Downgrade Strategy:
      1. Preserves ultra_fine regions with highest terrain complexity / changes.
      2. Preserves fine regions with high complexity.
      3. Preserves medium regions where affordable.
      4. Progressively downgrades lowest-complexity cells first:
         ultra_fine -> fine -> medium -> coarse.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        cost_model: Optional[ResolutionCostModel] = None,
        estimator: Optional[BudgetEstimator] = None,
    ):
        self.config = config or BudgetConfig()
        self.cost_model = cost_model or ResolutionCostModel()
        self.estimator = estimator or BudgetEstimator(cost_model=self.cost_model)

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "BudgetAwareResolutionOptimizer":
        """Instantiates optimizer directly from YAML configuration file."""
        cfg = BudgetConfig.from_yaml(config_path)
        return cls(config=cfg)

    def optimize(
        self,
        decision: ResolutionDecisionResult,
        terrain_result: Optional[TerrainAnalysisResult] = None,
        temporal_result: Optional[TemporalChangeResult] = None,
        config_override: Optional[BudgetConfig] = None,
    ) -> BudgetOptimizationResult:
        """
        Optimizes resolution assignments to adhere to compute and memory budgets.

        Args:
            decision: Initial ResolutionDecisionResult from AdaptiveResolutionSelector.
            terrain_result: Optional terrain features used to rank cell complexity.
            temporal_result: Optional temporal change detection used to boost changed cells.
            config_override: Optional BudgetConfig overriding instance config.

        Returns:
            BudgetOptimizationResult: Final optimized resolution plan with diagnostic stats.
        """
        cfg = config_override or self.config
        grid_map = decision.grid_map
        res_levels = decision.resolution_levels
        base_shape = grid_map.shape
        base_res = grid_map.resolution
        valid_mask = decision.valid_mask
        rows, cols = base_shape

        original_estimate = self.estimator.estimate_decision(decision)
        fits_original, original_violations = self.estimator.check_budget_fit(original_estimate, cfg)

        # ------------------------------------------------------------------
        # Early Exit: Budget already satisfied or budgeting disabled
        # ------------------------------------------------------------------
        if not cfg.enabled:
            return BudgetOptimizationResult(
                original_decision=decision,
                optimized_decision=decision,
                original_estimate=original_estimate,
                optimized_estimate=original_estimate,
                budget_config=cfg,
                original_fits_budget=fits_original,
                optimized_fits_budget=fits_original,
                is_downgraded=False,
                num_downgraded_cells=0,
                downgrade_counts_by_tier={},
                status="disabled",
                reasons=["Budgeting is disabled in configuration."],
            )

        if fits_original:
            return BudgetOptimizationResult(
                original_decision=decision,
                optimized_decision=decision,
                original_estimate=original_estimate,
                optimized_estimate=original_estimate,
                budget_config=cfg,
                original_fits_budget=True,
                optimized_fits_budget=True,
                is_downgraded=False,
                num_downgraded_cells=0,
                downgrade_counts_by_tier={},
                status="within_budget",
                reasons=[],
            )

        if not cfg.allow_resolution_downgrade:
            return BudgetOptimizationResult(
                original_decision=decision,
                optimized_decision=decision,
                original_estimate=original_estimate,
                optimized_estimate=original_estimate,
                budget_config=cfg,
                original_fits_budget=False,
                optimized_fits_budget=False,
                is_downgraded=False,
                num_downgraded_cells=0,
                downgrade_counts_by_tier={},
                status="impossible_budget",
                reasons=["Budget exceeded and allow_resolution_downgrade is False."] + original_violations,
            )

        # ------------------------------------------------------------------
        # Feasibility Check: Can an all-coarse map satisfy the budget?
        # ------------------------------------------------------------------
        num_decided = decision.num_decided_cells
        min_coarse_counts = {t: (num_decided if t == "coarse" else 0) for t in res_levels}
        min_estimate = self.estimator.estimate_from_counts(
            tier_counts=min_coarse_counts,
            resolution_levels=res_levels,
            base_shape=base_shape,
            base_resolution=base_res,
        )
        min_fits, min_violations = self.estimator.check_budget_fit(min_estimate, cfg)

        if not min_fits:
            # Even all-coarse exceeds configured budget!
            opt_assigned_level = np.where(valid_mask, "coarse", "")
            opt_assigned_res = np.where(valid_mask, res_levels.get("coarse", base_res), np.nan).astype(np.float32)
            opt_decision = ResolutionDecisionResult(
                grid_map=grid_map,
                terrain_result=decision.terrain_result,
                assigned_resolution=opt_assigned_res,
                assigned_level=opt_assigned_level,
                valid_mask=valid_mask,
                resolution_levels=res_levels,
                metadata={"budget_status": "impossible_budget", "violations": min_violations},
            )
            return BudgetOptimizationResult(
                original_decision=decision,
                optimized_decision=opt_decision,
                original_estimate=original_estimate,
                optimized_estimate=min_estimate,
                budget_config=cfg,
                original_fits_budget=False,
                optimized_fits_budget=False,
                is_downgraded=True,
                num_downgraded_cells=int(np.count_nonzero(valid_mask & (decision.assigned_level != "coarse"))),
                downgrade_counts_by_tier={"all_to_coarse": num_decided},
                status="impossible_budget",
                reasons=[f"Minimum coarse allocation ({min_estimate.memory_mb:.2f} MB, {min_estimate.allocated_cells:,} cells) exceeds budget."] + min_violations,
            )

        # ------------------------------------------------------------------
        # Compute Priority Ranking for each Grid Cell
        # ------------------------------------------------------------------
        priority_scores = np.zeros(base_shape, dtype=np.float32)

        if terrain_result is not None:
            # Composite terrain complexity score
            rough = np.nan_to_num(terrain_result.roughness, nan=0.0)
            slp = np.nan_to_num(terrain_result.slope, nan=0.0)
            el_rng = np.nan_to_num(terrain_result.elevation_range, nan=0.0)
            priority_scores = (rough * 3.0) + (slp * 1.5) + (el_rng * 1.0)
        else:
            # Fallback: elevation differences
            elev_diff = np.nan_to_num(grid_map.elevation_difference, nan=0.0)
            priority_scores = elev_diff.astype(np.float32)

        if temporal_result is not None:
            # Dynamically changed cells receive priority preservation boost
            changed_mask = temporal_result.changed_mask
            if np.any(changed_mask):
                priority_scores[changed_mask] *= cfg.temporal_change_weight

        # ------------------------------------------------------------------
        # Progressive Priority-Aware Downgrading Loop
        # ------------------------------------------------------------------
        opt_assigned_level = decision.assigned_level.copy()
        current_counts = decision.counts.copy()
        downgrade_counts_by_tier: Dict[str, int] = {t: 0 for t in ["ultra_fine", "fine", "medium"]}
        total_downgraded_cells = 0

        # Tiers to downgrade in descending order of cost
        tiers_to_downgrade = ["ultra_fine", "fine", "medium"]

        for tier in tiers_to_downgrade:
            target_lower_tier = self.cost_model.get_downgrade_target(tier)
            if target_lower_tier is None:
                continue

            # Find all cells currently in this tier
            tier_cell_mask = valid_mask & (opt_assigned_level == tier)
            cell_indices = np.argwhere(tier_cell_mask)

            if len(cell_indices) == 0:
                continue

            # Sort cells ascending by priority score (lowest complexity first)
            cell_scores = priority_scores[tier_cell_mask]
            sort_order = np.argsort(cell_scores)
            sorted_indices = cell_indices[sort_order]

            for r, c in sorted_indices:
                # Downgrade this cell to the next lower tier
                opt_assigned_level[r, c] = target_lower_tier
                current_counts[tier] -= 1
                current_counts[target_lower_tier] += 1
                downgrade_counts_by_tier[tier] += 1
                total_downgraded_cells += 1

                # Re-estimate resource usage
                cur_estimate = self.estimator.estimate_from_counts(
                    tier_counts=current_counts,
                    resolution_levels=res_levels,
                    base_shape=base_shape,
                    base_resolution=base_res,
                )
                fits, _ = self.estimator.check_budget_fit(cur_estimate, cfg)
                if fits:
                    break

            # Check if budget is satisfied after processing tier
            cur_estimate = self.estimator.estimate_from_counts(
                tier_counts=current_counts,
                resolution_levels=res_levels,
                base_shape=base_shape,
                base_resolution=base_res,
            )
            fits, _ = self.estimator.check_budget_fit(cur_estimate, cfg)
            if fits:
                break

        # ------------------------------------------------------------------
        # Construct Optimized Decision Result
        # ------------------------------------------------------------------
        opt_assigned_res = np.full(base_shape, np.nan, dtype=np.float32)
        for tier_name, res_val in res_levels.items():
            t_mask = valid_mask & (opt_assigned_level == tier_name)
            opt_assigned_res[t_mask] = res_val

        opt_metadata = dict(decision.metadata)
        opt_metadata.update({
            "budget_optimized": True,
            "total_downgrades": total_downgraded_cells,
            "downgrades_by_tier": downgrade_counts_by_tier,
        })

        optimized_decision = ResolutionDecisionResult(
            grid_map=grid_map,
            terrain_result=decision.terrain_result,
            assigned_resolution=opt_assigned_res,
            assigned_level=opt_assigned_level,
            valid_mask=valid_mask,
            resolution_levels=res_levels,
            metadata=opt_metadata,
        )

        optimized_estimate = self.estimator.estimate_decision(optimized_decision)
        opt_fits, opt_violations = self.estimator.check_budget_fit(optimized_estimate, cfg)

        status = "optimized_success" if opt_fits else "impossible_budget"
        active_downgrades = {k: v for k, v in downgrade_counts_by_tier.items() if v > 0}

        return BudgetOptimizationResult(
            original_decision=decision,
            optimized_decision=optimized_decision,
            original_estimate=original_estimate,
            optimized_estimate=optimized_estimate,
            budget_config=cfg,
            original_fits_budget=False,
            optimized_fits_budget=opt_fits,
            is_downgraded=(total_downgraded_cells > 0),
            num_downgraded_cells=total_downgraded_cells,
            downgrade_counts_by_tier=active_downgrades,
            status=status,
            reasons=original_violations if not opt_fits else [],
            metadata={"initial_violations": original_violations},
        )
