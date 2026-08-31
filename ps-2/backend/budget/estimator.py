"""
Budget and Resource Estimator.

Estimates allocated grid cells, memory footprint, and relative compute cost
from a resolution decision plan, and checks adherence to budget limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from backend.priority.result import ResolutionDecisionResult
from backend.budget.config import BudgetConfig
from backend.budget.cost_model import ResolutionCostModel, BYTES_PER_GRID_CELL


@dataclass
class ResourceEstimate:
    """
    Estimated resource requirements for a resolution decision plan.

    Attributes:
        allocated_cells: Total estimated allocated cells (tier fine cells + base index cells).
        memory_bytes: Total estimated array memory in bytes.
        memory_mb: Total estimated array memory in megabytes (MiB).
        compute_cost: Total estimated relative compute cost units.
        tier_counts: Base grid cell count assigned to each tier.
        tier_allocated_cells: Estimated fine-grid cells per tier.
        tier_memory_bytes: Estimated array bytes per tier.
        tier_compute_cost: Estimated compute units per tier.
        base_decision_cells: Total cells in the base decision grid.
    """
    allocated_cells: int
    memory_bytes: int
    memory_mb: float
    compute_cost: float
    tier_counts: Dict[str, int]
    tier_allocated_cells: Dict[str, int]
    tier_memory_bytes: Dict[str, int]
    tier_compute_cost: Dict[str, float]
    base_decision_cells: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocated_cells": self.allocated_cells,
            "memory_bytes": self.memory_bytes,
            "memory_mb": round(self.memory_mb, 4),
            "compute_cost": round(self.compute_cost, 2),
            "tier_counts": self.tier_counts,
            "tier_allocated_cells": self.tier_allocated_cells,
            "tier_memory_bytes": self.tier_memory_bytes,
            "tier_compute_cost": {k: round(v, 2) for k, v in self.tier_compute_cost.items()},
            "base_decision_cells": self.base_decision_cells,
        }


class BudgetEstimator:
    """
    Evaluates resource consumption and checks budget constraint satisfaction.
    """

    def __init__(self, cost_model: Optional[ResolutionCostModel] = None):
        self.cost_model = cost_model or ResolutionCostModel()

    def estimate_from_counts(
        self,
        tier_counts: Dict[str, int],
        resolution_levels: Dict[str, float],
        base_shape: Tuple[int, int],
        base_resolution: float = 0.20,
    ) -> ResourceEstimate:
        """
        Estimates resource requirements from tier cell counts.
        """
        base_rows, base_cols = base_shape
        base_total_cells = base_rows * base_cols

        tier_allocated: Dict[str, int] = {}
        tier_mem: Dict[str, int] = {}
        tier_comp: Dict[str, float] = {}

        total_tier_cells = 0
        total_tier_mem = 0
        total_comp = 0.0

        for tier, count in tier_counts.items():
            res = resolution_levels.get(tier, base_resolution)
            t_cells = self.cost_model.estimate_tier_cells(count, res, base_resolution)
            t_mem = self.cost_model.estimate_tier_memory_bytes(count, res, base_resolution)
            t_cost = self.cost_model.estimate_tier_compute_cost(count, res, base_resolution)

            tier_allocated[tier] = t_cells
            tier_mem[tier] = t_mem
            tier_comp[tier] = t_cost

            total_tier_cells += t_cells
            total_tier_mem += t_mem
            total_comp += t_cost

        # Base index matrices memory: index_row (int32, 4B) + index_col (int32, 4B) + index_tier (obj/str, ~16B)
        index_mem_bytes = base_total_cells * 24
        total_mem_bytes = total_tier_mem + index_mem_bytes
        total_allocated_cells = total_tier_cells + base_total_cells

        return ResourceEstimate(
            allocated_cells=total_allocated_cells,
            memory_bytes=total_mem_bytes,
            memory_mb=total_mem_bytes / (1024.0 * 1024.0),
            compute_cost=total_comp,
            tier_counts=dict(tier_counts),
            tier_allocated_cells=tier_allocated,
            tier_memory_bytes=tier_mem,
            tier_compute_cost=tier_comp,
            base_decision_cells=base_total_cells,
        )

    def estimate_decision(self, decision: ResolutionDecisionResult) -> ResourceEstimate:
        """
        Estimates resource requirements from a ResolutionDecisionResult.
        """
        counts = decision.counts
        res_levels = decision.resolution_levels
        base_shape = decision.grid_map.shape
        base_res = decision.grid_map.resolution
        return self.estimate_from_counts(
            tier_counts=counts,
            resolution_levels=res_levels,
            base_shape=base_shape,
            base_resolution=base_res,
        )

    @staticmethod
    def check_budget_fit(
        estimate: ResourceEstimate,
        config: BudgetConfig,
    ) -> Tuple[bool, List[str]]:
        """
        Validates whether a resource estimate satisfies configured budget limits.

        Returns:
            Tuple[bool, List[str]]: (fits_budget, list of violation reasons)
        """
        if not config.enabled:
            return True, []

        violations: List[str] = []

        if config.max_memory_mb is not None and estimate.memory_mb > config.max_memory_mb:
            violations.append(
                f"Memory limit exceeded: estimated {estimate.memory_mb:.2f} MB > max {config.max_memory_mb:.2f} MB"
            )

        if config.max_allocated_cells is not None and estimate.allocated_cells > config.max_allocated_cells:
            violations.append(
                f"Cell allocation limit exceeded: estimated {estimate.allocated_cells:,} cells > max {config.max_allocated_cells:,} cells"
            )

        if config.max_compute_cost is not None and estimate.compute_cost > config.max_compute_cost:
            violations.append(
                f"Compute cost limit exceeded: estimated {estimate.compute_cost:.1f} units > max {config.max_compute_cost:.1f} units"
            )

        fits = len(violations) == 0
        return fits, violations
