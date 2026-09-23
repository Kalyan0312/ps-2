#!/usr/bin/env python3
"""
Real-Time Adaptive LiDAR Streaming Demo (Phase 14 Part A)

Simulates a continuous stream of LiDAR frames passing through the RealTimeLidarPipeline.
Demonstrates initial map construction, stable frame reuse, incremental updates, and
full rebuild fallbacks.
"""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.mapping.config import MappingConfig
from backend.streaming.config import RealTimePipelineConfig
from backend.streaming.pipeline import RealTimeLidarPipeline


def main():
    print("=" * 60)
    print("REAL-TIME ADAPTIVE LIDAR STREAMING DEMO")
    print("=" * 60)

    # Initialize configs and pipeline
    map_cfg = MappingConfig.from_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    stream_cfg = RealTimePipelineConfig(max_buffer_size=10, collect_timing_metrics=True)
    pipeline = RealTimeLidarPipeline(config=stream_cfg, mapping_config=map_cfg)
    
    gen = SyntheticLidarGenerator(seed=100)

    # Pre-generate scenarios
    frames = []

    # Frame 0: Initial Map Construction
    pts0 = gen.generate_frame(frame_id=0, num_ground_points=4000).points
    frames.append(LidarFrame(points=pts0, frame_id=0, timestamp=0.0))

    # Frame 1: Identical / Mostly Stable
    frames.append(LidarFrame(points=pts0.copy(), frame_id=1, timestamp=0.1))

    # Frame 2: Small Localized Change (add a small object)
    pts2 = pts0.copy()
    mask2 = np.hypot(pts2[:, 0], pts2[:, 1]) < 1.5
    pts2[mask2, 2] += 0.7
    frames.append(LidarFrame(points=pts2, frame_id=2, timestamp=0.2))

    # Frame 3: Significant Scene Change (Highly Dynamic)
    pts3 = pts0.copy()
    pts3[:, 2] += 1.0 * np.sin(pts3[:, 0]) * np.cos(pts3[:, 1])
    frames.append(LidarFrame(points=pts3, frame_id=3, timestamp=0.3))

    # Frame 4: Another Stable Frame (Same as Frame 3)
    frames.append(LidarFrame(points=pts3.copy(), frame_id=4, timestamp=0.4))

    # Process Frames
    for frame in frames:
        print(f"\nFrame {frame.frame_id}")
        
        # We add a tiny sleep to simulate sensor delay if we wanted to test buffering, 
        # but for demo processing latency tracking we just feed them sequentially.
        
        result = pipeline.process_frame(frame)
        
        if result.is_initial_frame:
            status = "INITIAL_FRAME"
            strategy = "FULL_BUILD"
            changed_cells = result.adaptive_map.num_represented_cells
        else:
            status = "PROCESSED"
            strategy = result.selected_strategy
            if result.incremental_build_result:
                changed_cells = result.incremental_build_result.rebuilt_cells + result.incremental_build_result.invalidated_cells
            else:
                changed_cells = 0
                
        latency = result.metrics.get("total_latency_ms", 0.0)
        
        print(f"Status: {status}")
        print(f"Strategy: {strategy}")
        if not result.is_initial_frame:
            print(f"Changed Cells: {changed_cells}")
        print(f"Latency: {latency:.2f} ms")

    # Summary
    metrics = pipeline.get_performance_metrics()
    
    print("\n" + "=" * 60)
    print("STREAM SUMMARY")
    print("=" * 60)
    print(f"Frames Received:  {metrics['total_frames_received']}")
    print(f"Frames Processed: {metrics['total_frames_processed']}")
    print(f"Frames Dropped:   0") # Buffer not explicitly used in this sequential demo
    print("")
    print(f"Average Latency:  {metrics['average_latency_ms']:.2f} ms")
    print(f"Average FPS:      {metrics['average_fps']:.1f}")
    print("")
    
    for strat, count in metrics["strategy_counts"].items():
        print(f"{strat}: {count}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
