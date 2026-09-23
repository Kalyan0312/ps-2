"""
Recorded LiDAR Dataset Replay Source (Phase 17 Part B).

Implements BaseLidarSource by wrapping any BaseLidarLoader (KittiLidarLoader,
NumpyLidarLoader, SyntheticLidarLoader, or any future dataset loader) behind the
unified ingestion interface.

Architecture:
    BaseLidarLoader  (index-based, file-level)
          ↓
    DatasetReplaySource  (streaming, iterator-level)
          ↓
    BaseLidarSource  (get_next_frame / start / stop / is_active)
          ↓
    LidarFrameValidator
          ↓
    RealTimeLidarPipeline

Features:
    - Wraps any BaseLidarLoader implementation without modification.
    - Optional looping (replay dataset indefinitely until manually stopped).
    - Optional frame sub-range (start_index, max_frames).
    - Optional per-frame validation via LidarFrameValidator.
    - Lightweight: no extra buffering, no threads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List

from backend.data.base import BaseLidarLoader
from backend.data.frame import LidarFrame
from backend.ingestion.base import BaseLidarSource
from backend.ingestion.validator import LidarFrameValidator


class DatasetReplaySource(BaseLidarSource):
    """
    Recorded dataset replay source.

    Streams a pre-loaded or on-disk LiDAR dataset through the BaseLidarSource
    interface by sequentially indexing a BaseLidarLoader.

    Args:
        loader (BaseLidarLoader):
            Any BaseLidarLoader implementation (KittiLidarLoader, NumpyLidarLoader, etc.).
        start_index (int):
            Frame index at which playback begins (default 0).
        max_frames (Optional[int]):
            Maximum number of frames to yield before stopping. None = replay all frames.
        loop (bool):
            If True, restart from start_index after the last frame is yielded.
            Useful for continuous live-demo mode. Default False.
        validate_frames (bool):
            If True, pass each frame through LidarFrameValidator before yielding.
            Default True.
    """

    def __init__(
        self,
        loader: BaseLidarLoader,
        start_index: int = 0,
        max_frames: Optional[int] = None,
        loop: bool = False,
        validate_frames: bool = True,
    ) -> None:
        if not isinstance(loader, BaseLidarLoader):
            raise TypeError(
                f"DatasetReplaySource requires a BaseLidarLoader instance, "
                f"got {type(loader).__name__}."
            )

        total = len(loader)
        if total == 0:
            raise ValueError(
                "DatasetReplaySource received an empty loader (0 frames available). "
                "Ensure the dataset path contains valid LiDAR files."
            )

        if start_index < 0 or start_index >= total:
            raise IndexError(
                f"start_index={start_index} is out of range for loader with {total} frames."
            )

        self.loader = loader
        self.start_index = start_index
        self.max_frames = max_frames
        self.loop = loop
        self.validate_frames = validate_frames

        self._active: bool = False
        self._current_index: int = start_index
        self._frames_yielded: int = 0

    # ------------------------------------------------------------------
    # BaseLidarSource interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initializes the replay playhead at start_index and activates the source."""
        self._active = True
        self._current_index = self.start_index
        self._frames_yielded = 0

    def stop(self) -> None:
        """Deactivates the replay source. The current playhead position is preserved."""
        self._active = False

    def is_active(self) -> bool:
        """
        Returns True if the source is active and more frames are available.

        Becomes False when:
          - stop() was called, or
          - max_frames frames have been yielded, or
          - the loader is exhausted AND loop=False.
        """
        if not self._active:
            return False
        if self.max_frames is not None and self._frames_yielded >= self.max_frames:
            return False
        if not self.loop and self._current_index >= len(self.loader):
            return False
        return True

    def get_next_frame(self) -> Optional[LidarFrame]:
        """
        Retrieves the next frame from the dataset loader.

        Advances the internal playhead. When loop=True and the end of the dataset
        is reached, the playhead wraps back to start_index transparently.

        Returns:
            Optional[LidarFrame]: The next validated LiDAR frame, or None when
                the source is no longer active.
        """
        if not self.is_active():
            return None

        # Handle loop wrap-around
        total = len(self.loader)
        if self._current_index >= total:
            if self.loop:
                self._current_index = self.start_index
            else:
                return None

        frame = self.loader.get_frame(self._current_index)

        if self.validate_frames:
            frame = LidarFrameValidator.validate(frame)

        self._current_index += 1
        self._frames_yielded += 1
        return frame

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        """Total number of frames in the underlying dataset loader."""
        return len(self.loader)

    @property
    def frames_yielded(self) -> int:
        """Number of frames yielded since the last start() call."""
        return self._frames_yielded

    @property
    def current_index(self) -> int:
        """Current playhead index into the loader."""
        return self._current_index

    def __repr__(self) -> str:
        return (
            f"DatasetReplaySource("
            f"loader={type(self.loader).__name__}, "
            f"total_frames={self.total_frames}, "
            f"loop={self.loop}, "
            f"max_frames={self.max_frames})"
        )


def create_replay_source(
    source_path_or_files: Union[str, "Path", List[Union[str, "Path"]]],
    format_hint: Optional[str] = None,
    frame_rate_hz: float = 10.0,
    default_intensity: float = 0.5,
    start_index: int = 0,
    max_frames: Optional[int] = None,
    loop: bool = False,
    validate_frames: bool = True,
) -> DatasetReplaySource:
    """
    Factory that auto-detects the dataset format and returns a DatasetReplaySource.

    Supported formats:
      - ``.npy`` files (NumPy 2D arrays, Nx3 or Nx4)
      - ``.bin`` files (KITTI Velodyne float32 binary)
      - Explicit ``format_hint``: 'numpy'/'npy' or 'kitti'/'bin'

    Args:
        source_path_or_files: Directory path, single file path, or list of file paths.
        format_hint: Optional override for format detection ('numpy', 'npy', 'kitti', 'bin').
        frame_rate_hz: Sensor frame rate in Hz used for frame timestamps.
        default_intensity: Default intensity value when loading Nx3 NumPy files.
        start_index: Playback start index within the dataset.
        max_frames: Maximum frames to yield (None = all frames).
        loop: If True, replay loops indefinitely from start_index.
        validate_frames: If True, validate each frame at ingestion boundary.

    Returns:
        DatasetReplaySource: Ready-to-use replay source wrapping the auto-detected loader.

    Raises:
        FileNotFoundError: If the source path does not exist.
        ValueError: If the format is unsupported or no LiDAR files are found.
    """
    from backend.data.factory import create_lidar_loader

    loader = create_lidar_loader(
        source_path_or_files=source_path_or_files,
        format_hint=format_hint,
        frame_rate_hz=frame_rate_hz,
        default_intensity=default_intensity,
    )

    return DatasetReplaySource(
        loader=loader,
        start_index=start_index,
        max_frames=max_frames,
        loop=loop,
        validate_frames=validate_frames,
    )
