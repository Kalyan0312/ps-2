"""
Synthetic LiDAR Ingestion Source.

Implements BaseLidarSource by wrapping SyntheticLidarGenerator behind the unified
ingestion interface with boundary validation.
"""

from typing import Optional
from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator
from backend.ingestion.base import BaseLidarSource
from backend.ingestion.validator import LidarFrameValidator


class SyntheticLidarSource(BaseLidarSource):
    """
    Synthetic LiDAR frame source.
    Generates high-fidelity simulated ground and obstacle point cloud frames for testing and development.
    """

    def __init__(
        self,
        num_ground_points: int = 5000,
        frame_rate_hz: float = 15.0,
        max_frames: Optional[int] = None,
        seed: Optional[int] = 42,
        validate_frames: bool = True,
    ):
        self.num_ground_points = num_ground_points
        self.frame_rate_hz = max(0.1, frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz
        self.max_frames = max_frames
        self.validate_frames = validate_frames

        self.generator = SyntheticLidarGenerator(seed=seed)
        self._active = False
        self._frame_count = 0

    def start(self) -> None:
        """Starts the synthetic LiDAR ingestion source."""
        self._active = True
        self._frame_count = 0

    def stop(self) -> None:
        """Stops the synthetic LiDAR ingestion source."""
        self._active = False

    def is_active(self) -> bool:
        """Returns True if the source is active and frames remain to be yielded."""
        if not self._active:
            return False
        if self.max_frames is not None and self._frame_count >= self.max_frames:
            return False
        return True

    def get_next_frame(self) -> Optional[LidarFrame]:
        """
        Generates and validates the next synthetic LiDAR frame.

        Returns:
            Optional[LidarFrame]: Validated frame or None if stream is inactive / max_frames reached.
        """
        if not self.is_active():
            return None

        frame_id = self._frame_count
        timestamp = frame_id * self.dt

        frame = self.generator.generate_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            num_ground_points=self.num_ground_points,
        )

        if self.validate_frames:
            frame = LidarFrameValidator.validate(frame)

        self._frame_count += 1
        return frame
