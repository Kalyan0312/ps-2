"""
LiDAR Loader Factory and Auto-Detection Dispatcher.

Provides auto-detection for real LiDAR file formats (.npy, .bin) and unified
dataset loader instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Any, List

from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader
from backend.data.synthetic import SyntheticLidarLoader
from backend.data.numpy_loader import load_numpy_file, NumpyLidarLoader
from backend.data.kitti_loader import load_kitti_bin, KittiLidarLoader


def load_lidar_file(
    file_path: Union[str, Path],
    format_hint: Optional[str] = None,
    default_intensity: float = 0.5,
    frame_id: Any = None,
    timestamp: float = 0.0,
) -> LidarFrame:
    """
    Auto-detects and loads a single LiDAR point cloud file into a standard LidarFrame.

    Supported formats:
      - .npy: NumPy 2D array of shape (N, 3) or (N, 4)
      - .bin: KITTI Velodyne float32 binary file (4 floats per point)

    Args:
        file_path: Path to the LiDAR point cloud file.
        format_hint: Optional format override ('numpy', 'npy', 'kitti', 'bin').
        default_intensity: Default intensity value if Nx3 NumPy file is loaded.
        frame_id: Optional frame identifier.
        timestamp: Timestamp in seconds.

    Returns:
        LidarFrame: Point cloud with [x, y, z, intensity].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If format is unsupported or file contents are invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {path.resolve()}")

    fmt = format_hint.lower().strip() if format_hint else path.suffix.lower()

    if fmt in [".npy", "npy", "numpy"]:
        return load_numpy_file(
            file_path=path,
            default_intensity=default_intensity,
            frame_id=frame_id,
            timestamp=timestamp,
        )
    elif fmt in [".bin", "bin", "kitti", "velodyne"]:
        return load_kitti_bin(
            file_path=path,
            frame_id=frame_id,
            timestamp=timestamp,
        )
    else:
        raise ValueError(
            f"Unsupported LiDAR file format '{fmt}' for file '{path}'. "
            f"Supported formats: .npy (NumPy array), .bin (KITTI Velodyne binary)."
        )


def create_lidar_loader(
    source_path_or_files: Union[str, Path, List[Union[str, Path]]],
    format_hint: Optional[str] = None,
    frame_rate_hz: float = 10.0,
    default_intensity: float = 0.5,
) -> BaseLidarLoader:
    """
    Factory creating a BaseLidarLoader based on directory contents or format hints.

    Args:
        source_path_or_files: Directory path, file path, or list of file paths.
        format_hint: Explicit format ('synthetic', 'numpy', 'kitti').
        frame_rate_hz: Sensor frame rate in Hz.
        default_intensity: Default intensity for Nx3 NumPy files.

    Returns:
        BaseLidarLoader: Instantiated dataset loader.
    """
    if format_hint == "synthetic":
        return SyntheticLidarLoader(frame_rate_hz=frame_rate_hz)

    # Check if list of files
    if isinstance(source_path_or_files, (list, tuple)) and source_path_or_files:
        first_suffix = Path(source_path_or_files[0]).suffix.lower()
        if first_suffix == ".bin" or format_hint in ["kitti", "bin"]:
            return KittiLidarLoader(source_path_or_files, frame_rate_hz=frame_rate_hz)
        return NumpyLidarLoader(
            source_path_or_files,
            default_intensity=default_intensity,
            frame_rate_hz=frame_rate_hz,
        )

    path = Path(source_path_or_files)
    if path.is_dir():
        bin_files = list(path.glob("*.bin"))
        npy_files = list(path.glob("*.npy"))

        if format_hint in ["kitti", "bin"] or (bin_files and not npy_files):
            return KittiLidarLoader(path, frame_rate_hz=frame_rate_hz)
        elif format_hint in ["numpy", "npy"] or npy_files:
            return NumpyLidarLoader(
                path,
                default_intensity=default_intensity,
                frame_rate_hz=frame_rate_hz,
            )
        else:
            raise ValueError(
                f"Directory '{path}' does not contain recognized LiDAR files (.bin, .npy)."
            )
    elif path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".bin" or format_hint in ["kitti", "bin"]:
            return KittiLidarLoader([path], frame_rate_hz=frame_rate_hz)
        elif suffix == ".npy" or format_hint in ["numpy", "npy"]:
            return NumpyLidarLoader(
                [path],
                default_intensity=default_intensity,
                frame_rate_hz=frame_rate_hz,
            )
        else:
            raise ValueError(f"Unsupported file format '{suffix}' for '{path}'")
    else:
        raise FileNotFoundError(f"Source path not found: {path.resolve()}")
