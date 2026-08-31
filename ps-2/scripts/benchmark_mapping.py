#!/usr/bin/env python3
"""
Phase 7: Metrics, Memory Profiling, and Performance Benchmarking Script.

Executes and benchmarks:
1. Uniform-resolution 2.5D mapping (coarse, medium, fine, ultra-fine)
2. Multi-resolution mapping
3. Adaptive-resolution 2.5D mapping

Produces detailed timing statistics, spatial size evaluations, exact NumPy array
memory measurements, and fair comparative analysis.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.synthetic import SyntheticLidarGenerator
from backend.metrics.benchmark import BenchmarkEngine


def main():
    config_file = PROJECT_ROOT / "configs" / "config.yaml"
    
    # Initialize benchmark engine from config
    engine = BenchmarkEngine.from_config_file(config_file)
    
    # Generate sample frame
    raw_frame = SyntheticLidarGenerator(seed=42).generate_frame(
        frame_id=0,
        timestamp=0.0,
        num_ground_points=6000,
    )
    
    # Run complete benchmark suite
    result = engine.run(raw_frame)
    
    # Print formatted report
    print(result.formatted_report())


if __name__ == "__main__":
    main()
