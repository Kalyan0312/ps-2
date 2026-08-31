"""
Configuration for Metrics, Benchmarking, and Performance Profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union
import yaml


@dataclass
class BenchmarkConfig:
    """
    Configuration options for map benchmarking and timing.

    Attributes:
        enabled: If True, benchmarking and metrics collection are active.
        benchmark_runs: Number of repeated runs to average stage timing.
        warmup_runs: Number of initial warmup runs before recording timing.
        include_multires: Whether to evaluate multi-resolution map container.
    """
    enabled: bool = True
    benchmark_runs: int = 3
    warmup_runs: int = 1
    include_multires: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Constructs BenchmarkConfig from dictionary."""
        bench_data = data.get("benchmark", data.get("metrics", data))
        return cls(
            enabled=bool(bench_data.get("enabled", True)),
            benchmark_runs=max(1, int(bench_data.get("benchmark_runs", 3))),
            warmup_runs=max(0, int(bench_data.get("warmup_runs", 1))),
            include_multires=bool(bench_data.get("include_multires", True)),
        )

    @classmethod
    def from_yaml(cls, path: Union[Path, str]) -> "BenchmarkConfig":
        """Loads BenchmarkConfig directly from YAML file."""
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)
