"""
Metrics and Benchmarking module initialization.

Provides spatial size metrics, actual array memory profiling, timing utilities,
fair comparison analyzers, and benchmarking engines for 2.5D elevation grid maps.
"""

from backend.metrics.map_metrics import (
    MapSizeMetrics,
    MemoryMetrics,
    AdaptiveBreakdownMetrics,
    MapEvaluationMetrics,
    calculate_grid_map_metrics,
    calculate_multires_map_metrics,
    calculate_adaptive_map_metrics,
    calculate_map_metrics,
)
from backend.metrics.timing import (
    StageTiming,
    TimingMetrics,
    PipelineTimer,
)
from backend.metrics.comparison import (
    ComparisonMetric,
    ComparisonReport,
    compare_dense_allocation,
    compare_occupied_cells,
    compare_array_memory,
    build_fair_comparison_report,
)
from backend.metrics.config import BenchmarkConfig
from backend.metrics.benchmark import (
    BenchmarkResult,
    BenchmarkEngine,
)

__all__ = [
    "MapSizeMetrics",
    "MemoryMetrics",
    "AdaptiveBreakdownMetrics",
    "MapEvaluationMetrics",
    "calculate_grid_map_metrics",
    "calculate_multires_map_metrics",
    "calculate_adaptive_map_metrics",
    "calculate_map_metrics",
    "StageTiming",
    "TimingMetrics",
    "PipelineTimer",
    "ComparisonMetric",
    "ComparisonReport",
    "compare_dense_allocation",
    "compare_occupied_cells",
    "compare_array_memory",
    "build_fair_comparison_report",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkEngine",
]
