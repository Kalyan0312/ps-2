"""
Performance Timing Utilities and Metric Containers.

Provides high-precision stage timing using time.perf_counter(), supporting
single-run measurements, context managers, and multi-run statistical aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Iterator
from contextlib import contextmanager
import time
import numpy as np


@dataclass
class StageTiming:
    """
    Timing statistics for a single pipeline stage across one or more executions.

    Attributes:
        name: Stage identifier (e.g. 'preprocessing', 'adaptive_map_build').
        runs: Number of benchmark runs executed.
        total_time: Total elapsed time across all runs (seconds).
        mean_time: Mean execution time per run (seconds).
        min_time: Minimum execution time observed (seconds).
        max_time: Maximum execution time observed (seconds).
        std_time: Standard deviation of execution times (seconds).
        raw_times: List of elapsed times for each run (seconds).
    """
    name: str
    runs: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    std_time: float
    raw_times: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Mean execution time in milliseconds."""
        return self.mean_time * 1000.0

    @property
    def min_ms(self) -> float:
        """Minimum execution time in milliseconds."""
        return self.min_time * 1000.0

    @property
    def max_ms(self) -> float:
        """Maximum execution time in milliseconds."""
        return self.max_time * 1000.0

    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        return self.std_time * 1000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "mean_seconds": round(self.mean_time, 6),
            "total_seconds": round(self.total_time, 6),
        }

    @classmethod
    def from_samples(cls, name: str, samples: List[float]) -> "StageTiming":
        """Constructs StageTiming from a list of elapsed time measurements in seconds."""
        if not samples:
            return cls(
                name=name,
                runs=0,
                total_time=0.0,
                mean_time=0.0,
                min_time=0.0,
                max_time=0.0,
                std_time=0.0,
                raw_times=[],
            )
        arr = np.array(samples, dtype=np.float64)
        return cls(
            name=name,
            runs=len(samples),
            total_time=float(np.sum(arr)),
            mean_time=float(np.mean(arr)),
            min_time=float(np.min(arr)),
            max_time=float(np.max(arr)),
            std_time=float(np.std(arr)),
            raw_times=list(samples),
        )


@dataclass
class TimingMetrics:
    """
    Collection of stage timings and pipeline execution statistics.

    Attributes:
        stages: Dictionary mapping stage names to StageTiming objects.
        pipeline_total_mean_time: Sum of mean times across pipeline stages (seconds).
        benchmark_runs: Number of benchmark runs performed.
        warmup_runs: Number of warmup runs discarded.
    """
    stages: Dict[str, StageTiming] = field(default_factory=dict)
    pipeline_total_mean_time: float = 0.0
    benchmark_runs: int = 1
    warmup_runs: int = 0

    @property
    def pipeline_total_mean_ms(self) -> float:
        """Total pipeline mean execution time in milliseconds."""
        return self.pipeline_total_mean_time * 1000.0

    def get_stage(self, name: str) -> Optional[StageTiming]:
        """Retrieves StageTiming by name if present."""
        return self.stages.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_runs": self.benchmark_runs,
            "warmup_runs": self.warmup_runs,
            "pipeline_total_mean_ms": round(self.pipeline_total_mean_ms, 3),
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }


class PipelineTimer:
    """
    Timer utility that records high-resolution elapsed times using time.perf_counter().

    Supports:
    - Context manager timing: with timer.measure("stage"): ...
    - Manual start/stop: timer.start("stage"), timer.stop("stage")
    - Direct sample recording: timer.record("stage", elapsed)
    - Compiling into aggregated TimingMetrics across repeated runs.
    """

    def __init__(self):
        self._samples: Dict[str, List[float]] = {}
        self._active_starts: Dict[str, float] = {}

    def reset(self) -> None:
        """Clears all recorded samples."""
        self._samples.clear()
        self._active_starts.clear()

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        """Context manager measuring execution time of a code block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.record(stage_name, elapsed)

    def start(self, stage_name: str) -> None:
        """Starts timing a stage manually."""
        self._active_starts[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        """Stops timing a stage and records the elapsed time."""
        t1 = time.perf_counter()
        t0 = self._active_starts.pop(stage_name, None)
        if t0 is None:
            raise RuntimeError(f"Timer for stage '{stage_name}' was not started.")
        elapsed = t1 - t0
        self.record(stage_name, elapsed)
        return elapsed

    def record(self, stage_name: str, elapsed_seconds: float) -> None:
        """Directly appends an elapsed time sample for a stage."""
        if stage_name not in self._samples:
            self._samples[stage_name] = []
        self._samples[stage_name].append(float(elapsed_seconds))

    def get_samples(self, stage_name: str) -> List[float]:
        """Returns raw recorded samples for a stage."""
        return self._samples.get(stage_name, [])

    def compile(self, benchmark_runs: int = 1, warmup_runs: int = 0) -> TimingMetrics:
        """
        Compiles recorded samples into a TimingMetrics object.
        """
        stages: Dict[str, StageTiming] = {}
        total_mean = 0.0

        for stage_name, samples in self._samples.items():
            st = StageTiming.from_samples(stage_name, samples)
            stages[stage_name] = st
            total_mean += st.mean_time

        return TimingMetrics(
            stages=stages,
            pipeline_total_mean_time=total_mean,
            benchmark_runs=benchmark_runs,
            warmup_runs=warmup_runs,
        )
