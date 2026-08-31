"""
Core LiDAR Point Cloud Preprocessing Filter Functions.
Modular, vectorized operations on NumPy point cloud arrays with schema [x, y, z, intensity].
"""

from typing import Tuple
import numpy as np


def remove_invalid_points(points: np.ndarray) -> np.ndarray:
    """
    Removes NaN and Infinite values from point cloud.
    
    Args:
        points (np.ndarray): Shape (N, 4) with [x, y, z, intensity].
        
    Returns:
        np.ndarray: Cleaned points array without NaN / Inf.
    """
    if len(points) == 0:
        return points
    
    valid_mask = np.isfinite(points).all(axis=1)
    return points[valid_mask]


def filter_range(
    points: np.ndarray,
    min_range: float = 0.5,
    max_range: float = 25.0,
    use_2d: bool = False,
) -> np.ndarray:
    """
    Filters points based on radial distance from sensor origin (0, 0, 0).
    
    Args:
        points (np.ndarray): Shape (N, 4) with [x, y, z, intensity].
        min_range (float): Minimum distance cutoff in meters.
        max_range (float): Maximum distance cutoff in meters.
        use_2d (bool): If True, computes distance in XY plane; otherwise 3D Euclidean.
        
    Returns:
        np.ndarray: Points within [min_range, max_range].
    """
    if len(points) == 0:
        return points

    if use_2d:
        distances_sq = points[:, 0] ** 2 + points[:, 1] ** 2
    else:
        distances_sq = points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2

    min_sq = min_range ** 2
    max_sq = max_range ** 2

    mask = (distances_sq >= min_sq) & (distances_sq <= max_sq)
    return points[mask]


def filter_height(
    points: np.ndarray,
    min_z: float = -2.0,
    max_z: float = 4.0,
) -> np.ndarray:
    """
    Filters points based on vertical Z coordinate limits.
    
    Args:
        points (np.ndarray): Shape (N, 4) with [x, y, z, intensity].
        min_z (float): Minimum height limit in meters.
        max_z (float): Maximum height limit in meters.
        
    Returns:
        np.ndarray: Points where min_z <= z <= max_z.
    """
    if len(points) == 0:
        return points

    z_vals = points[:, 2]
    mask = (z_vals >= min_z) & (z_vals <= max_z)
    return points[mask]


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.1,
) -> np.ndarray:
    """
    Downsamples point cloud using a 3D regular voxel grid, computing cell centroids.
    
    Args:
        points (np.ndarray): Shape (N, 4) with [x, y, z, intensity].
        voxel_size (float): Voxel grid edge length in meters (must be > 0).
        
    Returns:
        np.ndarray: Downsampled point cloud with averaged coordinates and intensity.
    """
    if len(points) == 0:
        return points
    
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")

    # Compute discrete 3D voxel coordinates
    voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)

    # Group points by unique voxel coordinate
    _, inverse_indices, counts = np.unique(
        voxel_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    num_voxels = len(counts)
    sum_points = np.zeros((num_voxels, 4), dtype=np.float64)
    np.add.at(sum_points, inverse_indices, points.astype(np.float64))

    downsampled = (sum_points / counts[:, None]).astype(np.float32)
    return downsampled
