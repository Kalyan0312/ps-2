"""
Compute Budgeting and Resource-Aware Resolution Allocation module initialization.

Provides resource estimation, cost models, priority-preserving downgrading,
and budget-constrained resolution optimization.
"""

from backend.budget.config import BudgetConfig
from backend.budget.cost_model import ResolutionCostModel, DOWNGRADE_ORDER, BYTES_PER_GRID_CELL
from backend.budget.estimator import ResourceEstimate, BudgetEstimator
from backend.budget.result import BudgetOptimizationResult
from backend.budget.optimizer import BudgetAwareResolutionOptimizer

__all__ = [
    "BudgetConfig",
    "ResolutionCostModel",
    "DOWNGRADE_ORDER",
    "BYTES_PER_GRID_CELL",
    "ResourceEstimate",
    "BudgetEstimator",
    "BudgetOptimizationResult",
    "BudgetAwareResolutionOptimizer",
]
