"""
Fair Comparison and Efficiency Evaluation between Mapping Representations.

Implements rigorous, scientifically sound comparisons between Uniform grid maps
and AdaptiveMap25D across:
  A. Dense Spatial Allocation (grid cell count)
  B. Occupied Representation (cells populated with LiDAR data)
  C. Actual NumPy Array Memory (bytes measured from ndarrays)

Explicitly labels each comparison with its underlying metric and assumptions to
prevent conflating dense grid bounds with sparse occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from backend.metrics.map_metrics import MapEvaluationMetrics


@dataclass
class ComparisonMetric:
    """
    Detailed comparison between a baseline map representation and a target map representation.

    Attributes:
        name: Short descriptive title for the comparison.
        category: 'dense_allocation', 'occupied_cells', or 'array_memory'.
        baseline_name: Name of reference baseline (e.g. 'Uniform Ultra-Fine').
        baseline_value: Measured numeric value of baseline.
        target_name: Name of evaluated target (e.g. 'Adaptive Map').
        target_value: Measured numeric value of target.
        unit: Unit of measurement ('cells', 'bytes', 'KB', 'MB', etc.).
        difference: Difference (baseline_value - target_value). Positive indicates target is smaller.
        reduction_percentage: Percentage reduction relative to baseline ((baseline - target) / baseline * 100).
            Negative if target is larger.
        ratio: Target / Baseline ratio.
        is_reduction: True if target uses less resources than baseline.
        is_fair_comparison: True if comparison is apples-to-apples (e.g. array memory to array memory).
        description: Exact explanation of what is compared and the physical/computational meaning.
    """
    name: str
    category: str
    baseline_name: str
    baseline_value: float
    target_name: str
    target_value: float
    unit: str
    difference: float
    reduction_percentage: float
    ratio: float
    is_reduction: bool
    is_fair_comparison: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "baseline_name": self.baseline_name,
            "baseline_value": self.baseline_value,
            "target_name": self.target_name,
            "target_value": self.target_value,
            "unit": self.unit,
            "difference": round(self.difference, 2),
            "reduction_percentage": round(self.reduction_percentage, 2),
            "ratio": round(self.ratio, 4),
            "is_reduction": self.is_reduction,
            "is_fair_comparison": self.is_fair_comparison,
            "description": self.description,
        }


@dataclass
class ComparisonReport:
    """
    Container holding multi-dimensional fair comparison metrics between mapping approaches.

    Attributes:
        comparisons: List of ComparisonMetric instances.
    """
    comparisons: List[ComparisonMetric] = field(default_factory=list)

    def get_by_category(self, category: str) -> List[ComparisonMetric]:
        """Returns all comparison metrics within a specific category."""
        return [c for c in self.comparisons if c.category == category]

    def find(self, baseline_name: str, target_name: str, category: str) -> Optional[ComparisonMetric]:
        """Finds a specific comparison metric by baseline, target, and category."""
        for c in self.comparisons:
            if c.baseline_name == baseline_name and c.target_name == target_name and c.category == category:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "categories": {
                "dense_allocation": [c.to_dict() for c in self.get_by_category("dense_allocation")],
                "occupied_cells": [c.to_dict() for c in self.get_by_category("occupied_cells")],
                "array_memory": [c.to_dict() for c in self.get_by_category("array_memory")],
            }
        }


# ---------------------------------------------------------------------------
# Comparison builders
# ---------------------------------------------------------------------------

def compare_dense_allocation(
    baseline: MapEvaluationMetrics,
    target: MapEvaluationMetrics,
) -> ComparisonMetric:
    """
    Compares dense allocated grid cells.

    For Uniform grids: total allocated grid cells (rows * cols).
    For Adaptive grids: sum of allocated cells across tier bounding boxes + base decision grid cells.
    """
    base_val = float(baseline.size.total_cells)
    tgt_val = float(target.size.total_cells)
    diff = base_val - tgt_val
    reduction_pct = (diff / base_val * 100.0) if base_val > 0 else 0.0
    ratio = (tgt_val / base_val) if base_val > 0 else 0.0

    desc = (
        f"Compares total allocated grid array cells. Baseline '{baseline.name}' allocates "
        f"{int(base_val):,} cells; target '{target.name}' allocates {int(tgt_val):,} cells "
        f"(including tier bounding-box cells and index structure)."
    )

    return ComparisonMetric(
        name=f"Dense Allocation: {target.name} vs {baseline.name}",
        category="dense_allocation",
        baseline_name=baseline.name,
        baseline_value=base_val,
        target_name=target.name,
        target_value=tgt_val,
        unit="cells",
        difference=diff,
        reduction_percentage=reduction_pct,
        ratio=ratio,
        is_reduction=diff > 0,
        is_fair_comparison=True,
        description=desc,
    )


def compare_occupied_cells(
    baseline: MapEvaluationMetrics,
    target: MapEvaluationMetrics,
) -> ComparisonMetric:
    """
    Compares occupied cells containing actual LiDAR point observations.
    """
    base_val = float(baseline.size.occupied_cells)
    tgt_val = float(target.size.occupied_cells)
    diff = base_val - tgt_val
    reduction_pct = (diff / base_val * 100.0) if base_val > 0 else 0.0
    ratio = (tgt_val / base_val) if base_val > 0 else 0.0

    desc = (
        f"Compares occupied cells with point observations. Baseline '{baseline.name}' has "
        f"{int(base_val):,} occupied cells; target '{target.name}' has {int(tgt_val):,} occupied cells "
        f"across its active resolution tiers."
    )

    return ComparisonMetric(
        name=f"Occupied Representation: {target.name} vs {baseline.name}",
        category="occupied_cells",
        baseline_name=baseline.name,
        baseline_value=base_val,
        target_name=target.name,
        target_value=tgt_val,
        unit="occupied cells",
        difference=diff,
        reduction_percentage=reduction_pct,
        ratio=ratio,
        is_reduction=diff > 0,
        is_fair_comparison=True,
        description=desc,
    )


def compare_array_memory(
    baseline: MapEvaluationMetrics,
    target: MapEvaluationMetrics,
) -> ComparisonMetric:
    """
    Compares actual measured NumPy array memory in bytes.
    """
    base_val = float(baseline.memory.array_bytes)
    tgt_val = float(target.memory.array_bytes)
    diff = base_val - tgt_val
    reduction_pct = (diff / base_val * 100.0) if base_val > 0 else 0.0
    ratio = (tgt_val / base_val) if base_val > 0 else 0.0

    desc = (
        f"Compares exact NumPy array memory in bytes. Baseline '{baseline.name}' consumes "
        f"{base_val / 1024.0:.1f} KB ({int(base_val):,} bytes); target '{target.name}' consumes "
        f"{tgt_val / 1024.0:.1f} KB ({int(tgt_val):,} bytes) across all internal ndarrays."
    )

    return ComparisonMetric(
        name=f"Array Memory: {target.name} vs {baseline.name}",
        category="array_memory",
        baseline_name=baseline.name,
        baseline_value=base_val,
        target_name=target.name,
        target_value=tgt_val,
        unit="bytes",
        difference=diff,
        reduction_percentage=reduction_pct,
        ratio=ratio,
        is_reduction=diff > 0,
        is_fair_comparison=True,
        description=desc,
    )


def build_fair_comparison_report(
    uniform_evaluations: Dict[str, MapEvaluationMetrics],
    adaptive_evaluation: MapEvaluationMetrics,
) -> ComparisonReport:
    """
    Constructs a comprehensive comparison report comparing AdaptiveMap25D against
    each baseline Uniform grid map.
    """
    comparisons: List[ComparisonMetric] = []

    for name, uni_eval in uniform_evaluations.items():
        # 1. Dense spatial allocation comparison
        comparisons.append(compare_dense_allocation(baseline=uni_eval, target=adaptive_evaluation))
        # 2. Occupied representation comparison
        comparisons.append(compare_occupied_cells(baseline=uni_eval, target=adaptive_evaluation))
        # 3. Array memory comparison
        comparisons.append(compare_array_memory(baseline=uni_eval, target=adaptive_evaluation))

    return ComparisonReport(comparisons=comparisons)
