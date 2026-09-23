#!/usr/bin/env python3
"""
Recorded LiDAR Dataset Replay Demo (Phase 17 Part C)

Replays a recorded LiDAR dataset through the full adaptive mapping pipeline:

    Recorded .npy / .bin files
          ↓
    BaseLidarLoader  (NumpyLidarLoader or KittiLidarLoader)
          ↓
    DatasetReplaySource
          ↓
    LidarFrameValidator
          ↓
    LidarFrame
          ↓
    RealTimeLidarPipeline
          ↓
    AdaptiveMap25D

Usage (default fixture):
    python scripts/run_dataset_replay_demo.py

Usage (custom dataset):
    python scripts/run_dataset_replay_demo.py --dataset /path/to/npy_dir/
    python scripts/run_dataset_replay_demo.py --dataset /path/to/kitti/velodyne/ \\
        --max-frames 20 --start-index 5

CLI Options:
    --dataset       Path to directory or single file (.npy / .bin).
                    Default: tests/fixtures/lidar/ (built-in demo fixture).
    --max-frames    Maximum number of frames to process.
                    Default: process all available frames.
    --start-index   Frame index at which replay begins. Default: 0.
    --loop          If set, replay loops back to --start-index after the
                    last frame. Requires --max-frames to terminate.
    --no-validate   Skip per-frame validation (not recommended for production).
    --frame-rate    Sensor frame rate in Hz used to assign timestamps. Default: 10.0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of where this script is run from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.factory import create_lidar_loader
from backend.ingestion.replay import DatasetReplaySource, create_replay_source
from backend.mapping.config import MappingConfig
from backend.streaming.config import RealTimePipelineConfig
from backend.streaming.pipeline import RealTimeLidarPipeline

# Default fixture location — relative to project root
DEFAULT_FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "lidar"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_dataset_replay_demo.py",
        description="Replay a recorded LiDAR dataset through the adaptive mapping pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help=(
            "Path to a directory of .npy or .bin files, or a single .npy/.bin file. "
            "Defaults to the bundled demo fixture (tests/fixtures/lidar/)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N frames. Default: all available frames.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        metavar="IDX",
        help="Start replay from this frame index. Default: 0.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=False,
        help=(
            "Loop dataset indefinitely (must be combined with --max-frames to terminate)."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable per-frame validation (not recommended).",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=10.0,
        metavar="HZ",
        help="Sensor frame rate used for timestamp assignment. Default: 10.0 Hz.",
    )
    return parser


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _validate_args(args: argparse.Namespace) -> None:
    """Raise SystemExit with a clear message on invalid argument combinations."""

    if args.max_frames is not None and args.max_frames <= 0:
        print(
            f"[ERROR] --max-frames must be a positive integer, "
            f"got {args.max_frames}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.start_index < 0:
        print(
            f"[ERROR] --start-index must be >= 0, got {args.start_index}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.loop and args.max_frames is None:
        print(
            "[ERROR] --loop requires --max-frames to prevent infinite execution.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.frame_rate <= 0:
        print(
            f"[ERROR] --frame-rate must be > 0, got {args.frame_rate}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(
                f"[ERROR] Dataset path does not exist: {dataset_path.resolve()}",
                file=sys.stderr,
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# Resolve dataset path and create replay source
# ---------------------------------------------------------------------------

def _create_source(args: argparse.Namespace) -> DatasetReplaySource:
    """
    Resolves the dataset path (or default fixture) and creates a DatasetReplaySource.
    Exits with a clear error on any configuration or file problem.
    """
    dataset_path = Path(args.dataset) if args.dataset else DEFAULT_FIXTURE_DIR

    # Verify the default fixture is present
    if args.dataset is None and not dataset_path.exists():
        print(
            f"[ERROR] Built-in fixture dataset not found at: {dataset_path}\n"
            f"        Regenerate it with:\n"
            f"            python tests/fixtures/lidar/generate_fixtures.py",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        source = create_replay_source(
            source_path_or_files=dataset_path,
            frame_rate_hz=args.frame_rate,
            start_index=args.start_index,
            max_frames=args.max_frames,
            loop=args.loop,
            validate_frames=not args.no_validate,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except IndexError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    return source


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 60)
    print("RECORDED LIDAR DATASET REPLAY DEMO")
    print("Phase 17 Part C — Adaptive 2.5D Mapping Pipeline")
    print("=" * 60)

    dataset_path = Path(args.dataset) if args.dataset else DEFAULT_FIXTURE_DIR
    print(f"Dataset         : {dataset_path}")
    print(f"Start index     : {args.start_index}")
    print(f"Max frames      : {args.max_frames if args.max_frames else 'all'}")
    print(f"Loop            : {args.loop}")
    print(f"Frame rate      : {args.frame_rate} Hz")
    print(f"Validate frames : {not args.no_validate}")

    # ------------------------------------------------------------------
    # Create replay source
    # ------------------------------------------------------------------
    source = _create_source(args)
    print(f"Total frames in dataset  : {source.total_frames}")
    print()

    # ------------------------------------------------------------------
    # Pipeline setup (same config as run_realtime_stream_demo.py)
    # ------------------------------------------------------------------
    map_cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    if map_cfg_path.is_file():
        map_cfg = MappingConfig.from_yaml(map_cfg_path)
    else:
        map_cfg = MappingConfig()

    stream_cfg = RealTimePipelineConfig(max_buffer_size=10, collect_timing_metrics=True)
    pipeline = RealTimeLidarPipeline(config=stream_cfg, mapping_config=map_cfg)

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    frames_ok = 0
    frames_failed = 0
    t_demo_start = time.perf_counter()

    for frame in source:
        result = pipeline.process_frame(frame)

        latency_ms = result.metrics.get("total_latency_ms", 0.0)
        file_load_ms = result.metrics.get("preprocessing_time_ms", 0.0)
        map_ms = result.metrics.get("map_update_time_ms", 0.0)

        if result.success:
            frames_ok += 1
            strategy = result.selected_strategy if not result.is_initial_frame else "FULL_BUILD"
            represented = result.adaptive_map.num_represented_cells if result.adaptive_map else 0

            print(f"Frame {frame.frame_id}")
            print(f"  Timestamp        : {frame.timestamp:.3f} s")
            print(f"  Points           : {frame.num_points}")
            print(f"  Strategy         : {strategy}")
            print(f"  Adaptive cells   : {represented}")
            print(f"  File load + prep : {file_load_ms:.2f} ms")
            print(f"  Map update       : {map_ms:.2f} ms")
            print(f"  Total latency    : {latency_ms:.2f} ms")
            print()
        else:
            frames_failed += 1
            print(f"Frame {frame.frame_id}  [FAILED]  {result.error_message}")
            print()

    demo_elapsed_s = time.perf_counter() - t_demo_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    metrics = pipeline.get_performance_metrics()
    avg_latency = metrics["average_latency_ms"]
    avg_fps = metrics["average_fps"]
    total_received = metrics["total_frames_received"]

    print("=" * 60)
    print("REPLAY SUMMARY")
    print("=" * 60)
    print(f"Frames in dataset         : {source.total_frames}")
    print(f"Frames replayed           : {total_received}")
    print(f"Frames processed (OK)     : {frames_ok}")
    print(f"Frames failed             : {frames_failed}")
    print(f"Dropped frames            : 0  (synchronous replay)")
    print()
    print(f"Average mapping latency   : {avg_latency:.2f} ms")
    print(f"Average replay throughput : {avg_fps:.1f} FPS")
    print(f"Total demo wall time      : {demo_elapsed_s:.2f} s")
    print()

    for strat, count in metrics["strategy_counts"].items():
        print(f"  {strat:25s}: {count}")

    print("=" * 60)

    # Exit with non-zero code if any frame failed
    if frames_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
