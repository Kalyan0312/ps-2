"""
LiDAR Frame Data Model.
Represents a single point cloud frame with standard (x, y, z, intensity) schema.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class LidarFrame:
    """
    Standard LiDAR frame container.
    
    Attributes:
        points (np.ndarray): Array of shape (N, 4) with columns [x, y, z, intensity].
        frame_id (str | int): Identifier for the frame.
        timestamp (float): Timestamp in seconds.
        metadata (dict): Optional extra metadata (sensor model, pose, etc.).
    """
    points: np.ndarray
    frame_id: Any = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate points array
        if not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float32)

        if self.points.ndim != 2 or self.points.shape[1] != 4:
            raise ValueError(
                f"Expected points array of shape (N, 4) with [x, y, z, intensity], "
                f"got shape {self.points.shape}"
            )
        
        # Ensure float32 for high performance & standard precision
        if self.points.dtype != np.float32 and self.points.dtype != np.float64:
            self.points = self.points.astype(np.float32)

    @property
    def num_points(self) -> int:
        """Returns the total number of points in the frame."""
        return len(self.points)

    @property
    def xyz(self) -> np.ndarray:
        """Extracts the spatial coordinates (x, y, z) as shape (N, 3)."""
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        """Extracts the intensity channel as shape (N,)."""
        return self.points[:, 3]

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates spatial bounding box.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (min_xyz, max_xyz) each of shape (3,).
        """
        if self.num_points == 0:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        min_xyz = np.min(self.xyz, axis=0)
        max_xyz = np.max(self.xyz, axis=0)
        return min_xyz, max_xyz

    def get_summary(self) -> Dict[str, Any]:
        """Returns dictionary summary of the frame stats."""
        min_xyz, max_xyz = self.bounds
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "num_points": self.num_points,
            "shape": self.points.shape,
            "min_xyz": min_xyz.tolist(),
            "max_xyz": max_xyz.tolist(),
            "intensity_range": [
                float(np.min(self.intensity)) if self.num_points > 0 else 0.0,
                float(np.max(self.intensity)) if self.num_points > 0 else 0.0,
            ],
        }
