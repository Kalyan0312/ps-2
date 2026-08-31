"""
Benchmarking Engine and Comprehensive Result Containers.

Orchestrates multi-run performance profiling, spatial evaluation, memory measurement,
and fair comparisons between Uniform and Adaptive 2.5D elevation grid maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.multires_builder import MultiResolution25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig

from backend.metrics.config import BenchmarkConfig
from backend.metrics.timing import PipelineTimer, TimingMetrics
from backend.metrics.map_metrics import (
    calculate_grid_map_metrics,
    calculate_multires_map_metrics,
    calculate_adaptive_map_metrics,
    MapEvaluationMetrics,
)
from backend.metrics.comparison import (
    build_fair_comparison_report,
    ComparisonReport,
)


@dataclass
class BenchmarkResult:
    """
    Comprehensive benchmark evaluation results across all mapping representations.

    Attributes:
        input_raw_points: Number of points in input raw LiDAR frame.
        input_preprocessed_points: Number of points after preprocessing filtering.
        timing: TimingMetrics with statistical breakdown across benchmark runs.
        map_metrics: Dictionary of MapEvaluationMetrics keyed by representation name.
        comparisons: Fair comparison report between Uniform maps and Adaptive map.
        metadata: Execution parameters and environment metadata.
    """
    input_raw_points: int
    input_preprocessed_points: int
    timing: TimingMetrics
    map_metrics: Dict[str, MapEvaluationMetrics]
    comparisons: ComparisonReport
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_raw_points": self.input_raw_points,
            "input_preprocessed_points": self.input_preprocessed_points,
            "timing": self.timing.to_dict(),
            "map_metrics": {k: v.to_dict() for k, v in self.map_metrics.items()},
            "comparisons": self.comparisons.to_dict(),
            "metadata": self.metadata,
        }

    def get_efficiency_summary(self) -> str:
        """
        Generates an objective, scientifically accurate efficiency summary.
        """
        adaptive = self.map_metrics.get("Adaptive Map")
        uf = self.map_metrics.get("Uniform Ultra-Fine")
        coarse = self.map_metrics.get("Uniform Coarse")

        lines = [
            "EFFICIENCY SUMMARY:",
            "",
            "1. Cell Allocation & Spatial Footprint:",
        ]

        if coarse and adaptive:
            lines.append(
                f"   - '{coarse.name}' uses the fewest total allocated cells "
                f"({coarse.size.total_cells:,} cells @ {coarse.resolution:.2f} m/cell)."
            )
        if uf and adaptive:
            dense_comp = self.comparisons.find(uf.name, adaptive.name, "dense_allocation")
            if dense_comp:
                lines.append(
                    f"   - Dense Spatial Allocation: Adaptive Map allocates {adaptive.size.total_cells:,} "
                    f"cells vs Uniform Ultra-Fine's {uf.size.total_cells:,} cells "
                    f"({dense_comp.reduction_percentage:.1f}% reduction in dense array grid cells)."
                )

        lines.extend([
            "",
            "2. Array Memory Footprint:",
        ])
        if coarse and adaptive:
            lines.append(
                f"   - '{coarse.name}' uses the lowest array memory "
                f"({coarse.memory.array_kb:.1f} KB)."
            )
        if uf and adaptive:
            mem_comp = self.comparisons.find(uf.name, adaptive.name, "array_memory")
            if mem_comp:
                lines.append(
                    f"   - Actual Array Memory: Adaptive Map uses {adaptive.memory.array_kb:.1f} KB "
                    f"vs Uniform Ultra-Fine's {uf.memory.array_kb:.1f} KB "
                    f"({mem_comp.reduction_percentage:.1f}% array memory reduction)."
                )

        lines.extend([
            "",
            "3. Detail Preservation & Resolution Allocation:",
        ])
        if adaptive and adaptive.adaptive_breakdown:
            bd = adaptive.adaptive_breakdown
            lines.append(
                f"   - Preserves up to 0.05 m ultra-fine resolution in high-complexity terrain "
                f"({bd.tier_cell_counts.get('ultra_fine', 0):,} cells, "
                f"{bd.tier_percentages.get('ultra_fine', 0.0):.1f}% of represented area)."
            )
            lines.append(
                f"   - Stores benign/flat terrain compactly at coarse 0.50 m "
                f"({bd.tier_cell_counts.get('coarse', 0):,} cells, "
                f"{bd.tier_percentages.get('coarse', 0.0):.1f}% of represented area)."
            )

        lines.extend([
            "",
            "4. Fair Comparison & Scientific Integrity:",
            "   - Fair: Measured NumPy array memory compares actual ndarray.nbytes across all layers.",
            "   - Fair: Dense allocation compares full uniform grid bounds against adaptive tier bounding boxes.",
            "   - Note: Occupied cell count in adaptive mapping reflects multi-tier point binning, not empty space omission.",
        ])

        return "\n".join(lines)

    def formatted_report(self) -> str:
        """
        Formats benchmark results into a clean, human-readable terminal report.
        """
        lines = [
            "============================================================",
            "ADAPTIVE LiDAR MAPPING — PHASE 7 BENCHMARK",
            "============================================================",
            "",
            "Input:",
            f"  - Raw points          : {self.input_raw_points:,}",
            f"  - Preprocessed points : {self.input_preprocessed_points:,} "
            f"({(self.input_preprocessed_points / self.input_raw_points * 100.0) if self.input_raw_points > 0 else 0.0:.1f}% retained)",
            "",
            "TIMING:",
            f"{'Stage':<30} {'Average Time':>15} {'Min Time':>12} {'Max Time':>12}",
            "-" * 72,
        ]

        stage_order = [
            ("preprocessing", "Preprocessing"),
            ("uniform_coarse_build", "Uniform Coarse Build"),
            ("uniform_medium_build", "Uniform Medium Build"),
            ("uniform_fine_build", "Uniform Fine Build"),
            ("uniform_ultra_fine_build", "Uniform Ultra-Fine Build"),
            ("terrain_analysis", "Terrain Analysis"),
            ("resolution_decision", "Resolution Decision"),
            ("adaptive_map_build", "Adaptive Map Build"),
            ("multires_build", "Multi-Resolution Build"),
        ]

        for key, display_name in stage_order:
            st = self.timing.get_stage(key)
            if st:
                lines.append(f"{display_name:<30} {st.mean_ms:>12.2f} ms {st.min_ms:>9.2f} ms {st.max_ms:>9.2f} ms")

        # Total pipeline time
        total_st = self.timing.get_stage("total_adaptive_pipeline")
        if total_st:
            lines.append("-" * 72)
            lines.append(f"{'Total Adaptive Pipeline':<30} {total_st.mean_ms:>12.2f} ms {total_st.min_ms:>9.2f} ms {total_st.max_ms:>9.2f} ms")

        lines.extend([
            "",
            "MAP COMPARISON:",
            f"{'Representation':<22} {'Resolution':<12} {'Allocated Cells':>16} {'Occupied Cells':>16} {'Occupancy %':>12}",
            "-" * 80,
        ])

        for key in ["Uniform Coarse", "Uniform Medium", "Uniform Fine", "Uniform Ultra-Fine", "Adaptive Map", "Multi-Resolution Map"]:
            m = self.map_metrics.get(key)
            if m:
                res_str = f"{m.resolution:.2f} m" if m.resolution is not None else "Adaptive"
                if m.representation_type == "multires":
                    res_str = "Multi-Res"
                lines.append(
                    f"{m.name:<22} {res_str:<12} {m.size.total_cells:>16,} {m.size.occupied_cells:>16,} {m.size.occupancy_percentage:>11.1f}%"
                )

        adaptive_m = self.map_metrics.get("Adaptive Map")
        if adaptive_m and adaptive_m.adaptive_breakdown:
            abd = adaptive_m.adaptive_breakdown
            lines.append(
                f"  * Adaptive Map allocated cells: {abd.tier_allocated_cells:,} tier cells in dense bounding boxes + {abd.base_decision_cells:,} base index cells = {adaptive_m.size.total_cells:,} total storage cells."
            )

        lines.extend([
            "",
            "MEMORY COMPARISON:",
            f"{'Representation':<24} {'Actual Array Memory':>20} {'Estimated Total':>20}",
            "-" * 68,
        ])

        for key in ["Uniform Coarse", "Uniform Medium", "Uniform Fine", "Uniform Ultra-Fine", "Adaptive Map", "Multi-Resolution Map"]:
            m = self.map_metrics.get(key)
            if m:
                lines.append(
                    f"{m.name:<24} {m.memory.array_kb:>17.2f} KB {m.memory.estimated_object_bytes / 1024.0:>17.2f} KB"
                )

        lines.extend([
            "",
            self.get_efficiency_summary(),
            "============================================================",
        ])

        return "\n".join(lines)


class BenchmarkEngine:
    """
    Mapping Benchmark Engine.

    Executes the LiDAR mapping pipeline, measures performance timing across repeated runs,
    evaluates size and memory complexity, and produces fair comparison reports.
    """

    def __init__(
        self,
        mapping_config: Optional[MappingConfig] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
        terrain_config: Optional[TerrainConfig] = None,
        priority_config: Optional[PriorityConfig] = None,
        benchmark_config: Optional[BenchmarkConfig] = None,
    ):
        self.mapping_config = mapping_config or MappingConfig()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.terrain_config = terrain_config or TerrainConfig()
        self.priority_config = priority_config or PriorityConfig()
        self.benchmark_config = benchmark_config or BenchmarkConfig()

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "BenchmarkEngine":
        """Instantiates BenchmarkEngine directly from a YAML config file."""
        return cls(
            mapping_config=MappingConfig.from_yaml(config_path),
            preprocessing_config=PreprocessingConfig.from_yaml(config_path),
            terrain_config=TerrainConfig.from_yaml(config_path),
            priority_config=PriorityConfig.from_yaml(config_path),
            benchmark_config=BenchmarkConfig.from_yaml(config_path),
        )

    def run(self, frame: Optional[LidarFrame] = None) -> BenchmarkResult:
        """
        Executes benchmark suite on the provided or newly generated LiDAR frame.

        Args:
            frame: Optional LidarFrame to benchmark. If None, generates standard sample frame.

        Returns:
            BenchmarkResult: Complete benchmarking results.
        """
        raw_frame = frame or SyntheticLidarGenerator(seed=42).generate_frame(
            frame_id=0,
            timestamp=0.0,
            num_ground_points=6000,
        )
        raw_points_count = len(raw_frame.points)

        timer = PipelineTimer()
        runs = self.benchmark_config.benchmark_runs
        warmups = self.benchmark_config.warmup_runs
        total_executions = warmups + runs

        # Pipeline components
        preprocessor = PreprocessingPipeline(config=self.preprocessing_config)
        terrain_analyzer = TerrainAnalyzer(config=self.terrain_config)
        adaptive_selector = AdaptiveResolutionSelector(config=self.priority_config)
        adaptive_builder = AdaptiveMap25DBuilder(config=self.mapping_config)

        # Baseline uniform builders
        res_levels = self.mapping_config.multi_resolution_levels
        uniform_builders = {
            "coarse": Uniform25DMapBuilder(config=MappingConfig(
                resolution=res_levels.get("coarse", 0.50),
                min_x=self.mapping_config.min_x, max_x=self.mapping_config.max_x,
                min_y=self.mapping_config.min_y, max_y=self.mapping_config.max_y,
            )),
            "medium": Uniform25DMapBuilder(config=MappingConfig(
                resolution=res_levels.get("medium", 0.25),
                min_x=self.mapping_config.min_x, max_x=self.mapping_config.max_x,
                min_y=self.mapping_config.min_y, max_y=self.mapping_config.max_y,
            )),
            "fine": Uniform25DMapBuilder(config=MappingConfig(
                resolution=res_levels.get("fine", 0.10),
                min_x=self.mapping_config.min_x, max_x=self.mapping_config.max_x,
                min_y=self.mapping_config.min_y, max_y=self.mapping_config.max_y,
            )),
            "ultra_fine": Uniform25DMapBuilder(config=MappingConfig(
                resolution=res_levels.get("ultra_fine", 0.05),
                min_x=self.mapping_config.min_x, max_x=self.mapping_config.max_x,
                min_y=self.mapping_config.min_y, max_y=self.mapping_config.max_y,
            )),
        }

        # Multi-resolution builder
        multires_builder = MultiResolution25DMapBuilder(config=self.mapping_config)
        # Base builder for terrain analysis input
        base_builder = Uniform25DMapBuilder(config=self.mapping_config)

        # Variables to hold evaluated maps from final run
        last_preprocessed_frame: Optional[LidarFrame] = None
        last_uniform_maps: Dict[str, Any] = {}
        last_multires_map: Optional[Any] = None
        last_adaptive_map: Optional[Any] = None

        # Execute warmup + benchmark runs
        for exec_idx in range(total_executions):
            is_warmup = (exec_idx < warmups)

            # 1. Preprocessing
            t_pre_start = timer.start("preprocessing") if not is_warmup else None
            prep_frame = preprocessor.process(raw_frame)
            if not is_warmup:
                timer.stop("preprocessing")
            last_preprocessed_frame = prep_frame

            # 2. Uniform map builds
            for tier_name, builder in uniform_builders.items():
                stage_name = f"uniform_{tier_name}_build"
                if not is_warmup:
                    timer.start(stage_name)
                u_map = builder.build_map(prep_frame)
                if not is_warmup:
                    timer.stop(stage_name)
                last_uniform_maps[tier_name] = u_map

            # 3. Multi-resolution map build
            if self.benchmark_config.include_multires:
                if not is_warmup:
                    timer.start("multires_build")
                m_map = multires_builder.build_multires_map(prep_frame)
                if not is_warmup:
                    timer.stop("multires_build")
                last_multires_map = m_map

            # 4. Adaptive Pipeline Timing (Base grid + Terrain analysis + Decision + Adaptive map build)
            if not is_warmup:
                timer.start("total_adaptive_pipeline")

            base_grid = base_builder.build_map(prep_frame)

            if not is_warmup:
                timer.start("terrain_analysis")
            terrain_result = terrain_analyzer.analyze(base_grid)
            if not is_warmup:
                timer.stop("terrain_analysis")

            if not is_warmup:
                timer.start("resolution_decision")
            decision_result = adaptive_selector.select_resolution(base_grid, terrain_result)
            if not is_warmup:
                timer.stop("resolution_decision")

            if not is_warmup:
                timer.start("adaptive_map_build")
            adaptive_map = adaptive_builder.build(prep_frame, decision_result)
            if not is_warmup:
                timer.stop("adaptive_map_build")

            if not is_warmup:
                timer.stop("total_adaptive_pipeline")
            last_adaptive_map = adaptive_map

        # Compile timing metrics
        timing_metrics = timer.compile(benchmark_runs=runs, warmup_runs=warmups)

        # ------------------------------------------------------------------
        # Map complexity and memory evaluation
        # ------------------------------------------------------------------
        map_evaluations: Dict[str, MapEvaluationMetrics] = {}

        # Uniform evaluations
        name_map = {
            "coarse": "Uniform Coarse",
            "medium": "Uniform Medium",
            "fine": "Uniform Fine",
            "ultra_fine": "Uniform Ultra-Fine",
        }
        uniform_evals_for_comp: Dict[str, MapEvaluationMetrics] = {}
        for tier_name, u_map in last_uniform_maps.items():
            disp_name = name_map[tier_name]
            metrics = calculate_grid_map_metrics(u_map, name=disp_name)
            map_evaluations[disp_name] = metrics
            uniform_evals_for_comp[disp_name] = metrics

        # Multi-resolution evaluation
        if last_multires_map is not None:
            map_evaluations["Multi-Resolution Map"] = calculate_multires_map_metrics(
                last_multires_map, name="Multi-Resolution Map"
            )

        # Adaptive map evaluation
        assert last_adaptive_map is not None
        adaptive_eval = calculate_adaptive_map_metrics(
            last_adaptive_map, name="Adaptive Map"
        )
        map_evaluations["Adaptive Map"] = adaptive_eval

        # Fair comparisons report
        comparisons = build_fair_comparison_report(
            uniform_evaluations=uniform_evals_for_comp,
            adaptive_evaluation=adaptive_eval,
        )

        preprocessed_points_count = len(last_preprocessed_frame.points) if last_preprocessed_frame else 0

        return BenchmarkResult(
            input_raw_points=raw_points_count,
            input_preprocessed_points=preprocessed_points_count,
            timing=timing_metrics,
            map_metrics=map_evaluations,
            comparisons=comparisons,
            metadata={
                "benchmark_runs": runs,
                "warmup_runs": warmups,
                "resolutions": self.mapping_config.multi_resolution_levels,
                "base_resolution": self.mapping_config.resolution,
            },
        )
