"""
KITTI LiDAR Binary (.bin) Loader.

Loads raw float32 Velodyne LiDAR point clouds from KITTI-formatted .bin files
into standard LidarFrame instances [x, y, z, intensity/reflectance].

Format Specification:
---------------------
KITTI Velodyne scans are stored as binary files consisting of continuous float32 numbers:
  [x, y, z, reflectance]
Each point occupies exactly 16 bytes (4 float32 values * 4 bytes).

Sensor Coordinate System:
-------------------------
Points remain in the native KITTI Velodyne coordinate frame:
  - X: Forward
  - Y: Left
  - Z: Up
  - Reflectance: Scaled laser return intensity
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Any
import numpy as np

from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader


def load_kitti_bin(
    file_path: Union[str, Path],
    frame_id: Any = None,
    timestamp: float = 0.0,
) -> LidarFrame:
    """
    Loads a single KITTI Velodyne binary file (.bin) into a standard LidarFrame.

    Args:
        file_path: Path to the .bin file.
        frame_id: Frame identifier (defaults to filename stem if not specified).
        timestamp: Timestamp in seconds.

    Returns:
        LidarFrame: Point cloud with columns [x, y, z, reflectance].

    Raises:
        FileNotFoundError: If the binary file does not exist.
        ValueError: If file size is not a multiple of 16 bytes (4 float32 values per point).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"KITTI binary file not found: {path.resolve()}")

    file_size_bytes = path.stat().st_size

    # Validate 16-byte alignment (4 * 4 bytes)
    if file_size_bytes % 16 != 0:
        raise ValueError(
            f"Corrupted KITTI binary file '{path}': size {file_size_bytes} bytes "
            f"is not a multiple of 16 bytes (4 float32 values per point)."
        )

    # Empty file handling
    if file_size_bytes == 0:
        empty_points = np.empty((0, 4), dtype=np.float32)
        fid = frame_id if frame_id is not None else path.stem
        metadata = {
            "source": "kitti_bin",
            "file_path": str(path.resolve()),
            "file_format": ".bin",
            "original_point_count": 0,
            "sensor_coordinates": "kitti_velodyne",
        }
        return LidarFrame(
            points=empty_points,
            frame_id=fid,
            timestamp=timestamp,
            metadata=metadata,
        )

    # Read binary float32 data
    try:
        raw_data = np.fromfile(str(path), dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Failed to read binary data from '{path}': {exc}") from exc

    # Reshape to (N, 4)
    points = raw_data.reshape(-1, 4)
    num_points = len(points)
    fid = frame_id if frame_id is not None else path.stem

    metadata = {
        "source": "kitti_bin",
        "file_path": str(path.resolve()),
        "file_format": ".bin",
        "original_shape": points.shape,
        "original_point_count": num_points,
        "file_size_bytes": file_size_bytes,
        "sensor_coordinates": "kitti_velodyne",
    }

    return LidarFrame(
        points=points,
        frame_id=fid,
        timestamp=timestamp,
        metadata=metadata,
    )


class KittiLidarLoader(BaseLidarLoader):
    """
    Dataset loader for sequences of KITTI Velodyne .bin files.

    Indexes a directory (e.g., 'dataset/sequences/00/velodyne/') and loads frames sequentially.
    """

    def __init__(
        self,
        source_path_or_files: Union[str, Path, List[Union[str, Path]]],
        frame_rate_hz: float = 10.0,
    ):
        self.frame_rate_hz = max(0.001, frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz

        if isinstance(source_path_or_files, (list, tuple)):
            self.file_paths = [Path(p) for p in source_path_or_files if str(p).endswith(".bin")]
        else:
            root = Path(source_path_or_files)
            if root.is_dir():
                self.file_paths = sorted(root.glob("*.bin"))
            elif root.is_file() and root.suffix == ".bin":
                self.file_paths = [root]
            else:
                self.file_paths = []

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_frame(self, index: int) -> LidarFrame:
        if index < 0 or index >= len(self.file_paths):
            raise IndexError(f"Index {index} out of bounds for KITTI loader with {len(self.file_paths)} files.")

        path = self.file_paths[index]
        return load_kitti_bin(
            file_path=path,
            frame_id=index,
            timestamp=index * self.dt,
        )
