"""
Base LiDAR Loader Interface.
Defines an abstract loader for uniform frame ingestion across datasets.
"""

from abc import ABC, abstractmethod
from typing import Iterator
from backend.data.frame import LidarFrame


class BaseLidarLoader(ABC):
    """Abstract Base Class for LiDAR Dataset and Stream Loaders."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of frames available."""
        pass

    @abstractmethod
    def get_frame(self, index: int) -> LidarFrame:
        """
        Retrieves a single LiDAR frame by index.
        
        Args:
            index (int): Frame index.
            
        Returns:
            LidarFrame: Point cloud frame with x, y, z, intensity.
        """
        pass

    def __getitem__(self, index: int) -> LidarFrame:
        """Allows direct indexing (e.g., loader[0])."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range for loader of length {len(self)}")
        return self.get_frame(index)

    def __iter__(self) -> Iterator[LidarFrame]:
        """Allows standard iteration over dataset frames."""
        for i in range(len(self)):
            yield self.get_frame(i)
