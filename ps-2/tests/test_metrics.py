"""
Unit and Integration Tests for Phase 7: Metrics, Memory Profiling, and Benchmarking.
"""

import pytest
import numpy as np
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.mapping.grid_map import GridMap25D
from backend.mapping.multires_map import MultiResolutionMap
from backend.mapping.adaptive_map import AdaptiveMap25D
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.multires_builder import MultiResolution25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.mapping.config import MappingConfig
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.terrain.analyzer import TerrainAnalyzer
from backend.priority.selector import AdaptiveResolutionSelector

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_grid_map():
    rows, cols = 10, 20
    point_count = np.zeros((rows, cols), dtype=np.int64)
    min_z = np.full((rows, cols), np.nan, dtype=np.float64)
    max_z = np.full((rows, cols), np.nan, dtype=np.float64)
    mean_z = np.full((rows, cols), np.nan, dtype=np.float64)
    mean_intensity = np.full((rows, cols), np.nan, dtype=np.float64)

    # Populate 5 cells
    for r, c in [(0, 0), (1, 1), (2, 2), (5, 5), (9, 19)]:
        point_count[r, c] = 10
        min_z[r, c] = 0.5
        max_z[r, c] = 1.5
        mean_z[r, c] = 1.0
        mean_intensity[r, c] = 0.8

    return GridMap25D(
        min_x=-10.0,
        max_x=10.0,
        min_y=-5.0,
        max_y=5.0,
        resolution=1.0,
        point_count=point_count,
        min_z=min_z,
        max_z=max_z,
        mean_z=mean_z,
        mean_intensity=mean_intensity,
    )


@pytest.fixture
def sample_empty_grid_map():
    rows, cols = 5, 5
    return GridMap25D(
        min_x=0.0,
        max_x=5.0,
        min_y=0.0,
        max_y=5.0,
        resolution=1.0,
        point_count=np.zeros((rows, cols), dtype=np.int64),
        min_z=np.full((rows, cols), np.nan, dtype=np.float64),
        max_z=np.full((rows, cols), np.nan, dtype=np.float64),
        mean_z=np.full((rows, cols), np.nan, dtype=np.float64),
        mean_intensity=np.full((rows, cols), np.nan, dtype=np.float64),
    )


@pytest.fixture
def synthetic_frame():
    return SyntheticLidarGenerator(seed=42).generate_frame(
        frame_id=0,
        timestamp=0.0,
        num_ground_points=2000,
    )


# ---------------------------------------------------------------------------
# Test Map Size and Memory Metrics
# ---------------------------------------------------------------------------

class TestMapMetrics:

    def test_grid_map_metrics_calculation(self, sample_grid_map):
        metrics = calculate_grid_map_metrics(sample_grid_map, name="Test Uniform")
        assert metrics.name == "Test Uniform"
        assert metrics.representation_type == "uniform"
        assert metrics.resolution == 1.0
        assert metrics.size.rows == 10
        assert metrics.size.cols == 20
        assert metrics.size.total_cells == 200
        assert metrics.size.occupied_cells == 5
        assert metrics.size.empty_cells == 195
        assert pytest.approx(metrics.size.occupancy_percentage, 0.01) == 2.5
        assert metrics.size.dimensions == "10x20"

        # Check memory
        expected_bytes = (
            sample_grid_map.point_count.nbytes +
            sample_grid_map.min_z.nbytes +
            sample_grid_map.max_z.nbytes +
            sample_grid_map.mean_z.nbytes +
            sample_grid_map.mean_intensity.nbytes
        )
        assert metrics.memory.array_bytes == expected_bytes
        assert metrics.memory.array_kb == expected_bytes / 1024.0
        assert metrics.memory.is_actual_array_memory is True
        assert len(metrics.memory.layer_breakdown) == 5

    def test_empty_grid_map_metrics(self, sample_empty_grid_map):
        metrics = calculate_grid_map_metrics(sample_empty_grid_map)
        assert metrics.size.total_cells == 25
        assert metrics.size.occupied_cells == 0
        assert metrics.size.empty_cells == 25
        assert metrics.size.occupancy_percentage == 0.0
        assert metrics.memory.array_bytes > 0

    def test_multires_map_metrics(self, sample_grid_map, sample_empty_grid_map):
        mmap = MultiResolutionMap(
            maps={1.0: sample_grid_map, 0.5: sample_empty_grid_map},
            levels={"coarse": 1.0, "medium": 0.5},
        )
        metrics = calculate_multires_map_metrics(mmap, name="Test Multi-Res")
        assert metrics.representation_type == "multires"
        assert metrics.resolution is None
        assert metrics.size.total_cells == 200 + 25
        assert metrics.size.occupied_cells == 5 + 0
        assert metrics.size.empty_cells == 195 + 25
        assert metrics.memory.array_bytes > 0

    def test_polymorphic_dispatcher(self, sample_grid_map):
        metrics = calculate_map_metrics(sample_grid_map, name="Polymorphic Test")
        assert isinstance(metrics, MapEvaluationMetrics)
        assert metrics.name == "Polymorphic Test"

        with pytest.raises(TypeError):
            calculate_map_metrics("invalid_object")  # type: ignore

    def test_map_evaluation_to_dict(self, sample_grid_map):
        metrics = calculate_grid_map_metrics(sample_grid_map)
        d = metrics.to_dict()
        assert "name" in d
        assert "size" in d
        assert "memory" in d
        assert d["size"]["total_cells"] == 200
        assert d["memory"]["is_actual_array_memory"] is True


# ---------------------------------------------------------------------------
# Test Timing Metrics
# ---------------------------------------------------------------------------

class TestTimingMetrics:

    def test_stage_timing_from_samples(self):
        samples = [0.010, 0.012, 0.011, 0.009]
        st = StageTiming.from_samples("test_stage", samples)
        assert st.name == "test_stage"
        assert st.runs == 4
        assert pytest.approx(st.mean_time, 1e-4) == 0.0105
        assert pytest.approx(st.min_time, 1e-4) == 0.009
        assert pytest.approx(st.max_time, 1e-4) == 0.012
        assert pytest.approx(st.mean_ms, 1e-2) == 10.5
        assert st.std_time >= 0.0

    def test_stage_timing_empty_samples(self):
        st = StageTiming.from_samples("empty_stage", [])
        assert st.runs == 0
        assert st.mean_time == 0.0
        assert st.mean_ms == 0.0

    def test_pipeline_timer_context_and_manual(self):
        timer = PipelineTimer()
        with timer.measure("context_stage"):
            pass  # Fast execution

        timer.start("manual_stage")
        elapsed = timer.stop("manual_stage")
        assert elapsed >= 0.0

        timer.record("direct_stage", 0.05)

        compiled = timer.compile(benchmark_runs=1, warmup_runs=0)
        assert "context_stage" in compiled.stages
        assert "manual_stage" in compiled.stages
        assert "direct_stage" in compiled.stages
        assert compiled.stages["direct_stage"].mean_time == 0.05

    def test_pipeline_timer_unstarted_stop_raises(self):
        timer = PipelineTimer()
        with pytest.raises(RuntimeError):
            timer.stop("non_started_stage")


# ---------------------------------------------------------------------------
# Test Fair Comparison Logic
# ---------------------------------------------------------------------------

class TestComparisonLogic:

    def test_dense_allocation_comparison(self, sample_grid_map):
        base_metrics = calculate_grid_map_metrics(sample_grid_map, name="Baseline Uniform")
        # Target with 50 total cells
        target_metrics = MapEvaluationMetrics(
            name="Target Map",
            representation_type="adaptive",
            resolution=None,
            size=MapSizeMetrics(rows=5, cols=10, total_cells=50, occupied_cells=5, empty_cells=45, occupancy_percentage=10.0),
            memory=MemoryMetrics(array_bytes=500, estimated_object_bytes=1000, array_kb=0.5, array_mb=0.0005),
        )

        comp = compare_dense_allocation(baseline=base_metrics, target=target_metrics)
        assert comp.category == "dense_allocation"
        assert comp.baseline_value == 200
        assert comp.target_value == 50
        assert comp.difference == 150
        assert comp.reduction_percentage == 75.0
        assert comp.ratio == 0.25
        assert comp.is_reduction is True
        assert comp.is_fair_comparison is True

    def test_occupied_cells_comparison(self, sample_grid_map):
        base_metrics = calculate_grid_map_metrics(sample_grid_map, name="Baseline Uniform")
        target_metrics = calculate_grid_map_metrics(sample_grid_map, name="Target Map")

        comp = compare_occupied_cells(baseline=base_metrics, target=target_metrics)
        assert comp.category == "occupied_cells"
        assert comp.baseline_value == 5
        assert comp.target_value == 5
        assert comp.reduction_percentage == 0.0

    def test_array_memory_comparison(self, sample_grid_map):
        base_metrics = calculate_grid_map_metrics(sample_grid_map, name="Baseline")
        target_metrics = MapEvaluationMetrics(
            name="Target Map",
            representation_type="adaptive",
            resolution=None,
            size=MapSizeMetrics(rows=5, cols=10, total_cells=50, occupied_cells=5, empty_cells=45, occupancy_percentage=10.0),
            memory=MemoryMetrics(array_bytes=base_metrics.memory.array_bytes // 2, estimated_object_bytes=1000, array_kb=0.5, array_mb=0.0005),
        )

        comp = compare_array_memory(baseline=base_metrics, target=target_metrics)
        assert comp.category == "array_memory"
        assert comp.reduction_percentage == 50.0
        assert comp.is_reduction is True

    def test_build_fair_comparison_report(self, sample_grid_map):
        uni1 = calculate_grid_map_metrics(sample_grid_map, name="Uniform Coarse")
        uni2 = calculate_grid_map_metrics(sample_grid_map, name="Uniform Fine")
        adaptive = calculate_grid_map_metrics(sample_grid_map, name="Adaptive Map")

        report = build_fair_comparison_report({"Uniform Coarse": uni1, "Uniform Fine": uni2}, adaptive)
        assert len(report.comparisons) == 6  # 2 baselines * 3 categories
        assert len(report.get_by_category("dense_allocation")) == 2
        assert len(report.get_by_category("array_memory")) == 2
        assert len(report.get_by_category("occupied_cells")) == 2

        found = report.find("Uniform Coarse", "Adaptive Map", "dense_allocation")
        assert found is not None
        assert found.category == "dense_allocation"


# ---------------------------------------------------------------------------
# Test Benchmark Config
# ---------------------------------------------------------------------------

class TestBenchmarkConfig:

    def test_default_config(self):
        cfg = BenchmarkConfig()
        assert cfg.enabled is True
        assert cfg.benchmark_runs == 3
        assert cfg.warmup_runs == 1
        assert cfg.include_multires is True

    def test_from_dict(self):
        data = {
            "benchmark": {
                "enabled": True,
                "benchmark_runs": 5,
                "warmup_runs": 2,
                "include_multires": False,
            }
        }
        cfg = BenchmarkConfig.from_dict(data)
        assert cfg.benchmark_runs == 5
        assert cfg.warmup_runs == 2
        assert cfg.include_multires is False

    def test_from_yaml_file(self):
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            cfg = BenchmarkConfig.from_yaml(config_path)
            assert cfg.benchmark_runs >= 1
            assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Test Benchmark Engine End-to-End Pipeline
# ---------------------------------------------------------------------------

class TestBenchmarkEngineE2E:

    def test_engine_run(self, synthetic_frame):
        bench_cfg = BenchmarkConfig(
            enabled=True,
            benchmark_runs=2,
            warmup_runs=1,
            include_multires=True,
        )
        map_cfg = MappingConfig(
            resolution=0.5,
            min_x=-15.0, max_x=15.0,
            min_y=-15.0, max_y=15.0,
            multi_resolution_levels={
                "coarse": 0.50,
                "medium": 0.25,
                "fine": 0.10,
                "ultra_fine": 0.05,
            }
        )
        engine = BenchmarkEngine(
            mapping_config=map_cfg,
            benchmark_config=bench_cfg,
        )

        result = engine.run(synthetic_frame)

        assert isinstance(result, BenchmarkResult)
        assert result.input_raw_points == len(synthetic_frame.points)
        assert result.input_preprocessed_points > 0
        assert result.input_preprocessed_points <= result.input_raw_points

        # Verify all representations evaluated
        assert "Uniform Coarse" in result.map_metrics
        assert "Uniform Medium" in result.map_metrics
        assert "Uniform Fine" in result.map_metrics
        assert "Uniform Ultra-Fine" in result.map_metrics
        assert "Adaptive Map" in result.map_metrics
        assert "Multi-Resolution Map" in result.map_metrics

        # Verify timings recorded
        assert "preprocessing" in result.timing.stages
        assert "terrain_analysis" in result.timing.stages
        assert "resolution_decision" in result.timing.stages
        assert "adaptive_map_build" in result.timing.stages
        assert "total_adaptive_pipeline" in result.timing.stages

        # Verify adaptive map breakdown
        adaptive_metrics = result.map_metrics["Adaptive Map"]
        assert adaptive_metrics.adaptive_breakdown is not None
        assert adaptive_metrics.adaptive_breakdown.base_decision_cells > 0
        assert adaptive_metrics.adaptive_breakdown.tier_occupied_cells > 0

        # Verify formatted report and dict conversion
        report_text = result.formatted_report()
        assert "ADAPTIVE LiDAR MAPPING — PHASE 7 BENCHMARK" in report_text
        assert "TIMING:" in report_text
        assert "MAP COMPARISON:" in report_text
        assert "MEMORY COMPARISON:" in report_text
        assert "EFFICIENCY SUMMARY:" in report_text

        result_dict = result.to_dict()
        assert "timing" in result_dict
        assert "map_metrics" in result_dict
        assert "comparisons" in result_dict
