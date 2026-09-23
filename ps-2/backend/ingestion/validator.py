"""
Lightweight LiDAR Frame Boundary Validator.

Validates incoming LidarFrame objects at the ingestion boundary before they enter
the core mapping and streaming pipeline.
"""

from typing import Any
import numpy as np
from backend.data.frame import LidarFrame


class LidarFrameValidationError(ValueError):
    """Exception raised when a LiDAR frame fails ingestion boundary validation."""
    pass


class LidarFrameValidator:
    """
    Lightweight, high-performance boundary validator for LiDAR frames.
    Performs fast structural and numeric sanity checks without copying large point arrays.
    """

    @staticmethod
    def validate(frame: Any) -> LidarFrame:
        """
        Validates a LidarFrame instance.

        Args:
            frame (Any): Ingestion object to validate.

        Returns:
            LidarFrame: The validated frame instance.

        Raises:
            LidarFrameValidationError: If the frame is null, malformed, or contains invalid numeric data.
        """
        if frame is None:
            raise LidarFrameValidationError("LiDAR frame cannot be None.")

        if not isinstance(frame, LidarFrame):
            raise LidarFrameValidationError(
                f"Expected instance of LidarFrame, got {type(frame).__name__}."
            )

        pts = frame.points
        if not isinstance(pts, np.ndarray):
            raise LidarFrameValidationError("LidarFrame.points must be a numpy ndarray.")

        if pts.ndim != 2:
            raise LidarFrameValidationError(
                f"LidarFrame.points must be a 2D array, got ndim={pts.ndim} with shape {pts.shape}."
            )

        if pts.shape[1] < 3 or pts.shape[1] > 4:
            raise LidarFrameValidationError(
                f"LidarFrame.points must have 3 or 4 columns [x, y, z, (intensity)], got shape {pts.shape}."
            )

        if pts.shape[0] == 0:
            raise LidarFrameValidationError("LidarFrame.points cannot be empty (0 points).")

        # Check for non-finite values (NaN / Inf) efficiently
        if not np.all(np.isfinite(pts)):
            raise LidarFrameValidationError("LidarFrame.points contains non-finite values (NaN or Inf).")

        if not isinstance(frame.timestamp, (int, float)) or frame.timestamp < 0:
            raise LidarFrameValidationError(
                f"Invalid LidarFrame.timestamp '{frame.timestamp}'. Must be a non-negative float."
            )

        return frame
