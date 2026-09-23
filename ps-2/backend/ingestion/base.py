"""
Base LiDAR Ingestion Source Interface.

Defines the abstract interface for streaming or dataset-based LiDAR data sources
that yield canonical LidarFrame objects for pipeline consumption.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterator
from backend.data.frame import LidarFrame


class BaseLidarSource(ABC):
    """
    Abstract Base Class for LiDAR frame sources (synthetic generators,
    recorded dataset replays, ROS 2 topics, or hardware drivers).
    """

    @abstractmethod
    def start(self) -> None:
        """Initializes or opens the LiDAR data source."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Safely stops or closes the LiDAR data source."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Returns True if the data source is active and ready to yield frames."""
        pass

    @abstractmethod
    def get_next_frame(self) -> Optional[LidarFrame]:
        """
        Retrieves the next available LiDAR frame.

        Returns:
            Optional[LidarFrame]: Validated point cloud frame, or None if stream ended / unavailable.
        """
        pass

    def __iter__(self) -> Iterator[LidarFrame]:
        """Generator protocol allowing clean iteration over frames."""
        self.start()
        try:
            while self.is_active():
                frame = self.get_next_frame()
                if frame is None:
                    break
                yield frame
        finally:
            self.stop()
