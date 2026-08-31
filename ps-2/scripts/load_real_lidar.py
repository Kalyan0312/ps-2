#!/usr/bin/env python3
"""
Phase 8: Real LiDAR Point Cloud Inspection Script.

Loads and inspects real LiDAR point clouds from .npy and KITTI .bin files.

Usage:
  python scripts/load_real_lidar.py --file <path/to/file.bin|.npy>
  python scripts/load_real_lidar.py --file data/sample_kitti.bin
  python scripts/load_real_lidar.py --file data/sample_pointcloud.npy
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.factory import load_lidar_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect real LiDAR point cloud files (.npy, KITTI .bin)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to real LiDAR point cloud file (.npy or KITTI .bin)",
    )
    parser.add_argument(
        "--default-intensity",
        type=float,
        default=0.5,
        help="Default intensity if Nx3 NumPy array is supplied without intensity channel",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_file = args.file
    if target_file is None:
        # Check if default sample fixture exists
        sample_bin = PROJECT_ROOT / "data" / "sample_kitti.bin"
        if sample_bin.is_file():
            print("[*] No --file specified. Using built-in sample fixture: data/sample_kitti.bin")
            print("    To load your own dataset: python scripts/load_real_lidar.py --file /path/to/velodyne/000000.bin\n")
            target_file = str(sample_bin)
        else:
            print("============================================================")
            print("REAL LiDAR DATA LOADER — USAGE GUIDE")
            print("============================================================")
            print("No LiDAR file was provided. Please specify a file using --file:\n")
            print("1. KITTI Velodyne binary scan (.bin):")
            print("   python scripts/load_real_lidar.py --file /path/to/kitti/velodyne/000000.bin\n")
            print("2. NumPy point cloud (.npy) [Nx3 or Nx4]:")
            print("   python scripts/load_real_lidar.py --file /path/to/pointcloud.npy\n")
            print("============================================================")
            return

    path = Path(target_file)
    if not path.is_file():
        print(f"[!] Error: Specified file does not exist: {path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("============================================================")
    print("REAL LiDAR POINT CLOUD INSPECTION")
    print("============================================================")
    print(f"File path       : {path.resolve()}")
    print(f"File format     : {path.suffix.lower()}")
    print(f"File size       : {path.stat().st_size:,} bytes")

    try:
        frame = load_lidar_file(
            file_path=path,
            default_intensity=args.default_intensity,
        )
    except Exception as exc:
        print(f"[!] Failed to load LiDAR file: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = frame.get_summary()
    min_xyz, max_xyz = frame.bounds

    print("\n--- Point Cloud Properties ---")
    print(f"  Total points  : {frame.num_points:,}")
    print(f"  Array shape   : {frame.points.shape}")
    print(f"  Data dtype    : {frame.points.dtype}")

    print("\n--- Spatial Coordinate Extents ---")
    print(f"  X range (min/max) : [{min_xyz[0]:.3f} m, {max_xyz[0]:.3f} m]  (span: {max_xyz[0] - min_xyz[0]:.3f} m)")
    print(f"  Y range (min/max) : [{min_xyz[1]:.3f} m, {max_xyz[1]:.3f} m]  (span: {max_xyz[1] - min_xyz[1]:.3f} m)")
    print(f"  Z range (min/max) : [{min_xyz[2]:.3f} m, {max_xyz[2]:.3f} m]  (span: {max_xyz[2] - min_xyz[2]:.3f} m)")

    int_min, int_max = summary["intensity_range"]
    print("\n--- Intensity / Reflectance ---")
    print(f"  Intensity range   : [{int_min:.3f}, {int_max:.3f}]")

    print("\n--- Frame Metadata ---")
    for k, v in frame.metadata.items():
        print(f"  {k:<22}: {v}")

    print("============================================================")
    print("Successfully ingested into standard LidarFrame schema.")
    print("============================================================")


if __name__ == "__main__":
    main()
