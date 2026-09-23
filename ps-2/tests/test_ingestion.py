"""
Unit & Integration Tests for LiDAR Ingestion Subsystem (Phase 17 Part A).

Tests:
- BaseLidarSource interface compliance
- SyntheticLidarSource frame production
- LidarFrameValidator boundary validation
- Malformed frame rejection (null, wrong shape, NaNs, Infs)
- Pipeline consumption of ingestion source frames
- Start, stop, and iteration lifecycles
"""

import pytest
import numpy as np

from backend.data.frame import LidarFrame
from backend.ingestion import (
    BaseLidarSource,
    SyntheticLidarSource,
    LidarFrameValidator,
    LidarFrameValidationError,
)
from backend.streaming.pipeline import RealTimeLidarPipeline
from backend.streaming.config import RealTimePipelineConfig
from backend.mapping.config import MappingConfig


class TestLidarIngestionSubsystem:

    def test_synthetic_source_implements_base_interface(self):
        """1. Verify SyntheticLidarSource inherits from BaseLidarSource and implements abstract methods."""
        source = SyntheticLidarSource(max_frames=3)
        assert isinstance(source, BaseLidarSource)
        assert source.is_active() is False

        source.start()
        assert source.is_active() is True

        source.stop()
        assert source.is_active() is False

    def test_synthetic_source_produces_valid_frames(self):
        """2. Verify SyntheticLidarSource generates valid LidarFrame objects."""
        source = SyntheticLidarSource(num_ground_points=1000, max_frames=2, seed=123)
        source.start()

        frame0 = source.get_next_frame()
        assert frame0 is not None
        assert isinstance(frame0, LidarFrame)
        assert frame0.frame_id == 0
        assert frame0.timestamp == 0.0
        assert frame0.num_points > 1000
        assert frame0.points.shape[1] == 4

        frame1 = source.get_next_frame()
        assert frame1 is not None
        assert frame1.frame_id == 1
        assert frame1.timestamp > 0.0

        frame2 = source.get_next_frame()
        assert frame2 is None, "Expected None when max_frames limit reached"
        assert source.is_active() is False

    def test_validator_accepts_valid_frame(self):
        """3. Verify LidarFrameValidator accepts well-formed LidarFrame objects."""
        pts = np.array([
            [1.0, 2.0, 0.5, 100.0],
            [3.0, 4.0, 1.2, 110.0],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts, frame_id=1, timestamp=0.5)

        validated = LidarFrameValidator.validate(frame)
        assert validated is frame

    def test_validator_rejects_malformed_frames(self):
        """4. Verify LidarFrameValidator rejects malformed frames with descriptive LidarFrameValidationError."""
        # 4a. None frame
        with pytest.raises(LidarFrameValidationError, match="cannot be None"):
            LidarFrameValidator.validate(None)

        # 4b. Wrong class type
        with pytest.raises(LidarFrameValidationError, match="Expected instance of LidarFrame"):
            LidarFrameValidator.validate("not_a_frame")

        # 4c. Non-ndarray points
        frame_bad_points = LidarFrame(points=np.array([[0, 0, 0, 0]], dtype=np.float32), frame_id=0)
        frame_bad_points.points = "invalid_points_list"
        with pytest.raises(LidarFrameValidationError, match="must be a numpy ndarray"):
            LidarFrameValidator.validate(frame_bad_points)

        # 4d. 1D points array
        frame_1d = LidarFrame(points=np.array([[0, 0, 0, 0]], dtype=np.float32), frame_id=0)
        frame_1d.points = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        with pytest.raises(LidarFrameValidationError, match="must be a 2D array"):
            LidarFrameValidator.validate(frame_1d)

        # 4e. 0 points
        frame_empty = LidarFrame(points=np.array([[0, 0, 0, 0]], dtype=np.float32), frame_id=0)
        frame_empty.points = np.empty((0, 4), dtype=np.float32)
        with pytest.raises(LidarFrameValidationError, match="cannot be empty"):
            LidarFrameValidator.validate(frame_empty)

        # 4f. NaNs in points
        nan_pts = np.array([
            [1.0, 2.0, np.nan, 100.0],
        ], dtype=np.float32)
        frame_nan = LidarFrame(points=nan_pts, frame_id=0)
        with pytest.raises(LidarFrameValidationError, match="non-finite values"):
            LidarFrameValidator.validate(frame_nan)

        # 4g. Inf in points
        inf_pts = np.array([
            [1.0, np.inf, 0.5, 100.0],
        ], dtype=np.float32)
        frame_inf = LidarFrame(points=inf_pts, frame_id=0)
        with pytest.raises(LidarFrameValidationError, match="non-finite values"):
            LidarFrameValidator.validate(frame_inf)

        # 4h. Negative timestamp
        valid_pts = np.array([[1.0, 2.0, 0.5, 100.0]], dtype=np.float32)
        frame_neg_ts = LidarFrame(points=valid_pts, frame_id=0, timestamp=-5.0)
        with pytest.raises(LidarFrameValidationError, match="Invalid LidarFrame.timestamp"):
            LidarFrameValidator.validate(frame_neg_ts)

    def test_pipeline_consumes_ingestion_source_frames(self):
        """5. Verify RealTimeLidarPipeline consumes frames produced by SyntheticLidarSource."""
        map_cfg = MappingConfig(resolution=0.2, min_x=-25.0, max_x=25.0, min_y=-25.0, max_y=25.0)
        stream_cfg = RealTimePipelineConfig(max_buffer_size=10, collect_timing_metrics=True)
        pipeline = RealTimeLidarPipeline(config=stream_cfg, mapping_config=map_cfg)

        source = SyntheticLidarSource(num_ground_points=1000, max_frames=3, seed=42)

        results = []
        for frame in source:
            res = pipeline.process_frame(frame)
            results.append(res)

        assert len(results) == 3
        assert results[0].is_initial_frame is True
        assert results[0].success is True
        assert results[0].adaptive_map is not None
        assert results[1].success is True
        assert results[2].success is True

    def test_iteration_protocol(self):
        """6. Verify BaseLidarSource generator iteration protocol auto-starts and auto-stops."""
        source = SyntheticLidarSource(max_frames=4, seed=99)
        frames = list(source)

        assert len(frames) == 4
        assert source.is_active() is False
        assert [f.frame_id for f in frames] == [0, 1, 2, 3]
