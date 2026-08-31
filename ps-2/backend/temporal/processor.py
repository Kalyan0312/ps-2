"""
Temporal Stream and Sequence Processor.

Coordinates temporal change detection, state persistence, and multi-frame processing
across consecutive LiDAR frames or 2.5D grid maps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Any, Sequence

from backend.data.frame import LidarFrame
from backend.mapping.grid_map import GridMap25D
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.temporal.config import TemporalConfig
from backend.temporal.change_detector import TemporalChangeDetector
from backend.temporal.state import TemporalState
from backend.temporal.result import TemporalChangeResult


class TemporalProcessor:
    """
    High-level temporal processor for streaming LiDAR sequences.

    Maintains temporal state across consecutive frames, executes change detection,
    and tracks reusable stable spatial regions.
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.detector = TemporalChangeDetector(config=self.config)
        self.state = TemporalState()

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "TemporalProcessor":
        """Instantiates TemporalProcessor directly from YAML configuration file."""
        cfg = TemporalConfig.from_yaml(config_path)
        return cls(config=cfg)

    def reset(self) -> None:
        """Resets temporal tracking state."""
        self.state.reset()

    def process_grid_map(
        self,
        grid_map: GridMap25D,
        frame_id: Optional[Any] = None,
    ) -> TemporalChangeResult:
        """
        Processes a new GridMap25D frame against the accumulated temporal state.

        Args:
            grid_map: Current 2.5D elevation grid map.
            frame_id: Optional identifier for current frame.

        Returns:
            TemporalChangeResult: Masks, statistics, and stable regions.
        """
        cid = frame_id if frame_id is not None else grid_map.metadata.get("frame_id", self.state.frame_count)
        pid = self.state.last_frame_id

        # Run change detection against last recorded grid
        result = self.detector.detect_change(
            current_grid=grid_map,
            previous_grid=self.state.last_grid_map,
            frame_index=self.state.frame_count,
            current_frame_id=cid,
            previous_frame_id=pid,
        )

        # Update internal temporal state
        self.state.update(grid_map=grid_map, result=result)

        return result

    def process_frame(
        self,
        frame: LidarFrame,
        builder: Optional[Uniform25DMapBuilder] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
    ) -> TemporalChangeResult:
        """
        Processes a raw/preprocessed LidarFrame by constructing its 2.5D grid map
        and comparing it against prior temporal state.

        Args:
            frame: LiDAR point cloud frame.
            builder: Optional Uniform25DMapBuilder (defaults to default builder).
            preprocessor: Optional PreprocessingPipeline to filter points first.

        Returns:
            TemporalChangeResult: Temporal change detection results.
        """
        input_frame = preprocessor.process(frame) if preprocessor is not None else frame
        grid_builder = builder or Uniform25DMapBuilder()
        grid_map = grid_builder.build_map(input_frame)
        return self.process_grid_map(grid_map, frame_id=frame.frame_id)

    def process_sequence(
        self,
        frames: Sequence[LidarFrame],
        builder: Optional[Uniform25DMapBuilder] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
    ) -> List[TemporalChangeResult]:
        """
        Processes an entire sequence of LidarFrames chronologically.

        Args:
            frames: Sequence of LidarFrame instances.
            builder: Optional Uniform25DMapBuilder.
            preprocessor: Optional PreprocessingPipeline.

        Returns:
            List[TemporalChangeResult]: Results for each consecutive frame.
        """
        results: List[TemporalChangeResult] = []
        for frame in frames:
            res = self.process_frame(frame, builder=builder, preprocessor=preprocessor)
            results.append(res)
        return results
