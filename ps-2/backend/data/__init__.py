"""
Data module initialization.
Provides LiDAR frame representations, synthetic generators, and real dataset loaders (NumPy .npy, KITTI .bin).
"""

from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader
from backend.data.synthetic import SyntheticLidarGenerator, SyntheticLidarLoader
from backend.data.numpy_loader import load_numpy_file, NumpyLidarLoader
from backend.data.kitti_loader import load_kitti_bin, KittiLidarLoader
from backend.data.factory import load_lidar_file, create_lidar_loader

__all__ = [
    "LidarFrame",
    "BaseLidarLoader",
    "SyntheticLidarGenerator",
    "SyntheticLidarLoader",
    "load_numpy_file",
    "NumpyLidarLoader",
    "load_kitti_bin",
    "KittiLidarLoader",
    "load_lidar_file",
    "create_lidar_loader",
]
