"""
Map Size, Complexity, and Memory Evaluation Metrics.

Computes exact spatial allocation, occupancy statistics, and actual NumPy array
memory consumption for Uniform (GridMap25D), Multi-Resolution (MultiResolutionMap),
and Adaptive (AdaptiveMap25D) elevation maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from backend.mapping.grid_map import GridMap25D
from backend.mapping.multires_map import MultiResolutionMap
from backend.mapping.adaptive_map import AdaptiveMap25D


# ---------------------------------------------------------------------------
# Component metric dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MapSizeMetrics:
    """
    Spatial dimension and cell count metrics.

    Attributes:
        rows: Number of grid rows (along Y).
        cols: Number of grid columns (along X).
        total_cells: Total allocated grid cells in memory (rows * cols).
        occupied_cells: Count of cells containing >= 1 point.
        empty_cells: Count of cells containing 0 points.
        occupancy_percentage: Percentage of cells occupied [0.0%, 100.0%].
    """
    rows: int
    cols: int
    total_cells: int
    occupied_cells: int
    empty_cells: int
    occupancy_percentage: float

    @property
    def dimensions(self) -> str:
        """Formatted dimensions string 'RxC'."""
        return f"{self.rows}x{self.cols}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "dimensions": self.dimensions,
            "total_cells": self.total_cells,
            "occupied_cells": self.occupied_cells,
            "empty_cells": self.empty_cells,
            "occupancy_percentage": round(self.occupancy_percentage, 2),
        }


@dataclass
class MemoryMetrics:
    """
    Memory consumption metrics.

    Attributes:
        array_bytes: Sum of bytes consumed by the internal NumPy ndarrays.
        estimated_object_bytes: Estimated total memory including Python object overhead.
        array_kb: Array memory in kilobytes (KiB = bytes / 1024).
        array_mb: Array memory in megabytes (MiB = bytes / (1024 * 1024)).
        layer_breakdown: Breakdown of array bytes by layer/array name.
        is_actual_array_memory: True indicating bytes were measured directly from ndarray.nbytes.
    """
    array_bytes: int
    estimated_object_bytes: int
    array_kb: float
    array_mb: float
    layer_breakdown: Dict[str, int] = field(default_factory=dict)
    is_actual_array_memory: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "array_bytes": self.array_bytes,
            "estimated_object_bytes": self.estimated_object_bytes,
            "array_kb": round(self.array_kb, 3),
            "array_mb": round(self.array_mb, 4),
            "layer_breakdown": self.layer_breakdown,
            "is_actual_array_memory": self.is_actual_array_memory,
        }


@dataclass
class AdaptiveBreakdownMetrics:
    """
    Detailed breakdown specific to AdaptiveMap25D representation.

    Attributes:
        base_decision_cells: Total cells in the base decision grid evaluated (rows * cols).
        base_represented_cells: Count of base cells assigned to a resolution tier.
        base_unassigned_cells: Count of base cells with no tier assignment (empty).
        tier_allocated_cells: Sum of grid cells allocated across all per-tier GridMap25D instances.
        tier_occupied_cells: Sum of occupied cells across all per-tier GridMap25D instances.
        tier_empty_cells: Sum of empty cells across all per-tier GridMap25D instances.
        tier_cell_counts: Base cell count assigned to each tier name.
        tier_percentages: Percentage of represented base cells in each tier.
        per_tier_sizes: MapSizeMetrics for each individual tier grid.
        tier_array_bytes: Total array bytes in tier GridMap25D layers.
        index_array_bytes: Total array bytes in index matrices (index_tier, index_row, index_col).
    """
    base_decision_cells: int
    base_represented_cells: int
    base_unassigned_cells: int
    tier_allocated_cells: int
    tier_occupied_cells: int
    tier_empty_cells: int
    tier_cell_counts: Dict[str, int]
    tier_percentages: Dict[str, float]
    per_tier_sizes: Dict[str, MapSizeMetrics]
    tier_array_bytes: int
    index_array_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_decision_cells": self.base_decision_cells,
            "base_represented_cells": self.base_represented_cells,
            "base_unassigned_cells": self.base_unassigned_cells,
            "tier_allocated_cells": self.tier_allocated_cells,
            "tier_occupied_cells": self.tier_occupied_cells,
            "tier_empty_cells": self.tier_empty_cells,
            "tier_cell_counts": self.tier_cell_counts,
            "tier_percentages": {k: round(v, 2) for k, v in self.tier_percentages.items()},
            "per_tier_sizes": {k: v.to_dict() for k, v in self.per_tier_sizes.items()},
            "tier_array_bytes": self.tier_array_bytes,
            "index_array_bytes": self.index_array_bytes,
        }


@dataclass
class MapEvaluationMetrics:
    """
    Comprehensive evaluation metrics for a single map representation.

    Attributes:
        name: Human-readable map representation name (e.g. 'Uniform Ultra-Fine', 'Adaptive Map').
        representation_type: 'uniform', 'multires', or 'adaptive'.
        resolution: Grid resolution in meters, or None for multi-resolution / adaptive maps.
        size: MapSizeMetrics object containing dimension and cell counts.
        memory: MemoryMetrics object containing array and object byte measurements.
        adaptive_breakdown: Optional detailed breakdown for adaptive maps.
        metadata: Construction or frame metadata dictionary.
    """
    name: str
    representation_type: str
    resolution: Optional[float]
    size: MapSizeMetrics
    memory: MemoryMetrics
    adaptive_breakdown: Optional[AdaptiveBreakdownMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "representation_type": self.representation_type,
            "resolution": self.resolution,
            "size": self.size.to_dict(),
            "memory": self.memory.to_dict(),
        }
        if self.adaptive_breakdown is not None:
            result["adaptive_breakdown"] = self.adaptive_breakdown.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# Metric calculation helper functions
# ---------------------------------------------------------------------------

def calculate_grid_map_metrics(
    grid: GridMap25D,
    name: str = "Uniform Map",
    representation_type: str = "uniform",
) -> MapEvaluationMetrics:
    """
    Calculates size and memory metrics for a single GridMap25D instance.

    Array memory is measured by summing `.nbytes` of all 5 grid layers:
    point_count, min_z, max_z, mean_z, mean_intensity.
    """
    rows, cols = grid.shape
    total_cells = grid.total_cells
    occupied_cells = grid.num_occupied_cells
    empty_cells = grid.num_empty_cells
    occupancy_pct = grid.occupancy_percentage

    size = MapSizeMetrics(
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        occupied_cells=occupied_cells,
        empty_cells=empty_cells,
        occupancy_percentage=occupancy_pct,
    )

    layers: Dict[str, int] = {
        "point_count": int(grid.point_count.nbytes),
        "min_z": int(grid.min_z.nbytes),
        "max_z": int(grid.max_z.nbytes),
        "mean_z": int(grid.mean_z.nbytes),
        "mean_intensity": int(grid.mean_intensity.nbytes),
    }
    array_bytes = sum(layers.values())
    # Estimate Python wrapper object overhead (~256 bytes per dataclass + dict overhead)
    estimated_object_bytes = array_bytes + 512

    memory = MemoryMetrics(
        array_bytes=array_bytes,
        estimated_object_bytes=estimated_object_bytes,
        array_kb=array_bytes / 1024.0,
        array_mb=array_bytes / (1024.0 * 1024.0),
        layer_breakdown=layers,
        is_actual_array_memory=True,
    )

    return MapEvaluationMetrics(
        name=name,
        representation_type=representation_type,
        resolution=float(grid.resolution),
        size=size,
        memory=memory,
        metadata=dict(grid.metadata),
    )


def calculate_multires_map_metrics(
    mmap: MultiResolutionMap,
    name: str = "Multi-Resolution Map",
) -> MapEvaluationMetrics:
    """
    Calculates combined size and memory metrics for a MultiResolutionMap.
    """
    total_cells = 0
    occupied_cells = 0
    empty_cells = 0
    layer_breakdown: Dict[str, int] = {}
    total_array_bytes = 0

    for res, grid in mmap.maps.items():
        lvl_name = mmap.get_level_name(res) or f"{res:.2f}m"
        total_cells += grid.total_cells
        occupied_cells += grid.num_occupied_cells
        empty_cells += grid.num_empty_cells

        grid_bytes = (
            grid.point_count.nbytes +
            grid.min_z.nbytes +
            grid.max_z.nbytes +
            grid.mean_z.nbytes +
            grid.mean_intensity.nbytes
        )
        layer_breakdown[f"grid_{lvl_name}_{res:.2f}m"] = int(grid_bytes)
        total_array_bytes += int(grid_bytes)

    occupancy_pct = (occupied_cells / total_cells * 100.0) if total_cells > 0 else 0.0

    # MultiResolutionMap doesn't have a single row/col dimension, use total counts
    size = MapSizeMetrics(
        rows=len(mmap),
        cols=0,
        total_cells=total_cells,
        occupied_cells=occupied_cells,
        empty_cells=empty_cells,
        occupancy_percentage=occupancy_pct,
    )

    memory = MemoryMetrics(
        array_bytes=total_array_bytes,
        estimated_object_bytes=total_array_bytes + (len(mmap) * 512) + 256,
        array_kb=total_array_bytes / 1024.0,
        array_mb=total_array_bytes / (1024.0 * 1024.0),
        layer_breakdown=layer_breakdown,
        is_actual_array_memory=True,
    )

    return MapEvaluationMetrics(
        name=name,
        representation_type="multires",
        resolution=None,
        size=size,
        memory=memory,
        metadata=dict(mmap.metadata),
    )


def calculate_adaptive_map_metrics(
    amap: AdaptiveMap25D,
    name: str = "Adaptive Map",
) -> MapEvaluationMetrics:
    """
    Calculates detailed metrics for an AdaptiveMap25D.

    Clearly separates:
    - Base decision cells (cells in the reference decision grid)
    - Represented base cells
    - Actual tier-grid allocated cells (sum of rows*cols in each tier's tight GridMap25D)
    - Occupied adaptive cells (sum of occupied cells across tier grids)
    - Exact NumPy array memory: tier GridMap25D layers + index matrices (index_tier, index_row, index_col)
    """
    base_rows, base_cols = amap.base_shape
    base_decision_cells = base_rows * base_cols
    base_represented_cells = amap.num_represented_cells
    base_unassigned_cells = base_decision_cells - base_represented_cells

    tier_allocated_cells = 0
    tier_occupied_cells = 0
    tier_empty_cells = 0
    per_tier_sizes: Dict[str, MapSizeMetrics] = {}
    layer_breakdown: Dict[str, int] = {}
    tier_array_bytes = 0

    for tier_name, tier_grid in amap.tier_maps.items():
        t_tot = tier_grid.total_cells
        t_occ = tier_grid.num_occupied_cells
        t_emp = tier_grid.num_empty_cells
        t_pct = tier_grid.occupancy_percentage

        tier_allocated_cells += t_tot
        tier_occupied_cells += t_occ
        tier_empty_cells += t_emp

        per_tier_sizes[tier_name] = MapSizeMetrics(
            rows=tier_grid.rows,
            cols=tier_grid.cols,
            total_cells=t_tot,
            occupied_cells=t_occ,
            empty_cells=t_emp,
            occupancy_percentage=t_pct,
        )

        t_bytes = (
            tier_grid.point_count.nbytes +
            tier_grid.min_z.nbytes +
            tier_grid.max_z.nbytes +
            tier_grid.mean_z.nbytes +
            tier_grid.mean_intensity.nbytes
        )
        layer_breakdown[f"tier_grid_{tier_name}"] = int(t_bytes)
        tier_array_bytes += int(t_bytes)

    # Index matrices memory
    row_idx_bytes = int(amap.index_row.nbytes)
    col_idx_bytes = int(amap.index_col.nbytes)
    # Object array nbytes stores pointer array; add approximate string bytes for strings
    tier_idx_pointer_bytes = int(amap.index_tier.nbytes)
    # Estimate ~48 bytes per unique/assigned string object in Python
    tier_idx_str_bytes = int(np.count_nonzero(amap.index_tier != "")) * 48
    index_array_bytes = row_idx_bytes + col_idx_bytes + tier_idx_pointer_bytes + tier_idx_str_bytes

    layer_breakdown["index_row"] = row_idx_bytes
    layer_breakdown["index_col"] = col_idx_bytes
    layer_breakdown["index_tier"] = tier_idx_pointer_bytes + tier_idx_str_bytes

    total_array_bytes = tier_array_bytes + index_array_bytes
    estimated_object_bytes = total_array_bytes + (len(amap.tier_maps) * 512) + 1024

    breakdown = AdaptiveBreakdownMetrics(
        base_decision_cells=base_decision_cells,
        base_represented_cells=base_represented_cells,
        base_unassigned_cells=base_unassigned_cells,
        tier_allocated_cells=tier_allocated_cells,
        tier_occupied_cells=tier_occupied_cells,
        tier_empty_cells=tier_empty_cells,
        tier_cell_counts=amap.tier_cell_counts,
        tier_percentages=amap.tier_percentages,
        per_tier_sizes=per_tier_sizes,
        tier_array_bytes=tier_array_bytes,
        index_array_bytes=index_array_bytes,
    )

    # Total allocated cells in the adaptive structure includes base index cells + tier allocated cells
    total_allocated_adaptive = tier_allocated_cells + base_decision_cells
    occupancy_pct = (
        (tier_occupied_cells / tier_allocated_cells * 100.0)
        if tier_allocated_cells > 0 else 0.0
    )

    size = MapSizeMetrics(
        rows=base_rows,
        cols=base_cols,
        total_cells=total_allocated_adaptive,
        occupied_cells=tier_occupied_cells,
        empty_cells=tier_empty_cells,
        occupancy_percentage=occupancy_pct,
    )

    memory = MemoryMetrics(
        array_bytes=total_array_bytes,
        estimated_object_bytes=estimated_object_bytes,
        array_kb=total_array_bytes / 1024.0,
        array_mb=total_array_bytes / (1024.0 * 1024.0),
        layer_breakdown=layer_breakdown,
        is_actual_array_memory=True,
    )

    return MapEvaluationMetrics(
        name=name,
        representation_type="adaptive",
        resolution=None,
        size=size,
        memory=memory,
        adaptive_breakdown=breakdown,
        metadata=dict(amap.metadata),
    )


def calculate_map_metrics(
    map_obj: Union[GridMap25D, MultiResolutionMap, AdaptiveMap25D],
    name: Optional[str] = None,
) -> MapEvaluationMetrics:
    """
    Polymorphic dispatcher calculating metrics for any supported 2.5D map type.
    """
    if isinstance(map_obj, GridMap25D):
        return calculate_grid_map_metrics(map_obj, name=name or "Uniform Map")
    elif isinstance(map_obj, MultiResolutionMap):
        return calculate_multires_map_metrics(map_obj, name=name or "Multi-Resolution Map")
    elif isinstance(map_obj, AdaptiveMap25D):
        return calculate_adaptive_map_metrics(map_obj, name=name or "Adaptive Map")
    else:
        raise TypeError(f"Unsupported map type for metric calculation: {type(map_obj)}")
