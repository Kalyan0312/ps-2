"""
Budget Optimization Result Data Structure.

Encapsulates original vs budget-constrained resolution allocations, resource usage deltas,
downgrade distributions, and budget satisfaction diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from backend.priority.result import ResolutionDecisionResult
from backend.budget.config import BudgetConfig
from backend.budget.estimator import ResourceEstimate


@dataclass
class BudgetOptimizationResult:
    """
    Result of compute budgeting and resource-aware resolution allocation.

    Attributes:
        original_decision: Input ResolutionDecisionResult before budget constraints.
        optimized_decision: Final ResolutionDecisionResult after budget optimization.
        original_estimate: ResourceEstimate for original resolution plan.
        optimized_estimate: ResourceEstimate for final optimized resolution plan.
        budget_config: BudgetConfig used during optimization.
        original_fits_budget: True if original plan already satisfied all budget constraints.
        optimized_fits_budget: True if final optimized plan satisfies all budget constraints.
        is_downgraded: True if at least one cell was downgraded to meet budget constraints.
        num_downgraded_cells: Total count of base cells that were downgraded.
        downgrade_counts_by_tier: Count of cells downgraded FROM each tier (e.g. {'ultra_fine': 10}).
        status: Optimization status ('within_budget', 'optimized_success', 'impossible_budget', 'disabled').
        reasons: List of explanatory strings describing budget violations or actions taken.
        metadata: Additional diagnostic metadata.
    """
    original_decision: ResolutionDecisionResult
    optimized_decision: ResolutionDecisionResult
    original_estimate: ResourceEstimate
    optimized_estimate: ResourceEstimate
    budget_config: BudgetConfig
    original_fits_budget: bool
    optimized_fits_budget: bool
    is_downgraded: bool
    num_downgraded_cells: int
    downgrade_counts_by_tier: Dict[str, int]
    status: str
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_counts(self) -> Dict[str, int]:
        """Cell counts per tier in original resolution plan."""
        return self.original_decision.counts

    @property
    def optimized_counts(self) -> Dict[str, int]:
        """Cell counts per tier in optimized resolution plan."""
        return self.optimized_decision.counts

    @property
    def downgrade_percentage(self) -> float:
        """Percentage of decided cells that were downgraded [0.0%, 100.0%]."""
        total = self.original_decision.num_decided_cells
        if total == 0:
            return 0.0
        return (self.num_downgraded_cells / total) * 100.0

    @property
    def memory_reduction_mb(self) -> float:
        """Memory reduction in MB achieved by optimization."""
        return max(0.0, self.original_estimate.memory_mb - self.optimized_estimate.memory_mb)

    @property
    def memory_reduction_percentage(self) -> float:
        """Percentage reduction in estimated array memory."""
        orig = self.original_estimate.memory_mb
        if orig <= 0:
            return 0.0
        return (self.memory_reduction_mb / orig) * 100.0

    def get_cell(self, row: int, col: int) -> Dict[str, Any]:
        """Returns cell diagnostic comparing original vs optimized tier assignment."""
        orig_info = self.original_decision.get_cell(row, col)
        opt_level = str(self.optimized_decision.assigned_level[row, col])
        opt_res = float(self.optimized_decision.assigned_resolution[row, col]) if opt_level else None

        orig_level = orig_info.get("assigned_level")
        was_downgraded = bool(orig_level and opt_level and orig_level != opt_level)

        return {
            "row": row,
            "col": col,
            "original_level": orig_level,
            "optimized_level": opt_level if opt_level else None,
            "original_resolution": orig_info.get("assigned_resolution"),
            "optimized_resolution": opt_res,
            "was_downgraded": was_downgraded,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive structured summary of budget optimization."""
        return {
            "status": self.status,
            "original_fits_budget": self.original_fits_budget,
            "optimized_fits_budget": self.optimized_fits_budget,
            "is_downgraded": self.is_downgraded,
            "num_downgraded_cells": self.num_downgraded_cells,
            "downgrade_percentage": round(self.downgrade_percentage, 2),
            "downgrade_counts_by_tier": self.downgrade_counts_by_tier,
            "original_counts": self.original_counts,
            "optimized_counts": self.optimized_counts,
            "original_memory_mb": round(self.original_estimate.memory_mb, 4),
            "optimized_memory_mb": round(self.optimized_estimate.memory_mb, 4),
            "memory_reduction_mb": round(self.memory_reduction_mb, 4),
            "memory_reduction_pct": round(self.memory_reduction_percentage, 2),
            "original_allocated_cells": self.original_estimate.allocated_cells,
            "optimized_allocated_cells": self.optimized_estimate.allocated_cells,
            "reasons": self.reasons,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_summary()
