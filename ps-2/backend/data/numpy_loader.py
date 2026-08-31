"""
NumPy Point Cloud Loader.

Loads point clouds from .npy binary files into standard LidarFrame instances.
Supports Nx3 (x, y, z) and Nx4 (x, y, z, intensity) point clouds with configurable
default intensity for 3D coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Any, Iterator
import numpy as np

from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader


def load_numpy_file(
    file_path: Union[str, Path],
    default_intensity: float = 0.5,
    frame_id: Any = None,
    timestamp: float = 0.0,
) -> LidarFrame:
    """
    Loads a single point cloud from a .npy file into a standard LidarFrame.

    Supports:
      - (N, 3) arrays: columns [x, y, z] -> appended with default_intensity
      - (N, 4) arrays: columns [x, y, z, intensity] -> preserved as-is

    Args:
        file_path: Path to the .npy file.
        default_intensity: Default intensity value in [0.0, 1.0] when loading Nx3 points.
        frame_id: Frame identifier (defaults to filename stem if not specified).
        timestamp: Frame timestamp in seconds.

    Returns:
        LidarFrame: Standard (N, 4) point cloud container.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If array dimensions or columns are invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"NumPy point cloud file not found: {path.resolve()}")

    try:
        raw_arr = np.load(path)
    except Exception as exc:
        raise ValueError(f"Failed to read NumPy file '{path}': {exc}") from exc

    if not isinstance(raw_arr, np.ndarray):
        raise ValueError(f"Expected np.ndarray from '{path}', got {type(raw_arr)}")

    # Check dimensions
    if raw_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D point cloud array (N, 3) or (N, 4), got shape {raw_arr.shape} in '{path}'"
        )

    num_cols = raw_arr.shape[1]
    original_shape = raw_arr.shape
    orig_point_count = len(raw_arr)

    if num_cols == 3:
        # Append default intensity
        xyz = raw_arr.astype(np.float32)
        intensity = np.full((orig_point_count, 1), fill_value=default_intensity, dtype=np.float32)
        points = np.hstack([xyz, intensity])
    elif num_cols == 4:
        points = raw_arr.astype(np.float32)
    else:
        raise ValueError(
            f"Invalid number of point cloud columns: {num_cols}. "
            f"Expected 3 ([x, y, z]) or 4 ([x, y, z, intensity]) in '{path}'"
        )

    fid = frame_id if frame_id is not None else path.stem

    metadata = {
        "source": "numpy",
        "file_path": str(path.resolve()),
        "file_format": ".npy",
        "original_shape": original_shape,
        "original_point_count": orig_point_count,
        "has_native_intensity": (num_cols == 4),
        "default_intensity_used": (num_cols == 3),
        "sensor_coordinates": "cartesian",
    }

    return LidarFrame(
        points=points,
        frame_id=fid,
        timestamp=timestamp,
        metadata=metadata,
    )


class NumpyLidarLoader(BaseLidarLoader):
    """
    Dataset loader for sequences of .npy point cloud files.

    Indexes a directory or list of .npy file paths and provides frame-by-frame loading.
    """

    def __init__(
        self,
        source_path_or_files: Union[str, Path, List[Union[str, Path]]],
        default_intensity: float = 0.5,
        frame_rate_hz: float = 10.0,
    ):
        self.default_intensity = default_intensity
        self.frame_rate_hz = max(0.001, frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz

        if isinstance(source_path_or_files, (list, tuple)):
            self.file_paths = [Path(p) for p in source_path_or_files if str(p).endswith(".npy")]
        else:
            root = Path(source_path_or_files)
            if root.is_dir():
                self.file_paths = sorted(root.glob("*.npy"))
            elif root.is_file() and root.suffix == ".npy":
                self.file_paths = [root]
            else:
                self.file_paths = []

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_frame(self, index: int) -> LidarFrame:
        if index < 0 or index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of bounds for loader with {len(self.file_paths)} files.")

        path = self.file_paths[index]
        return load_numpy_file(
            file_path=path,
            default_intensity=self.default_intensity,
            frame_id=index,
            timestamp=index * self.dt,
        )
