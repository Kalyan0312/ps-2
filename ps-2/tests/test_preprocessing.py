"""
Unit tests for LiDAR Point Cloud Preprocessing (Phase 2).
"""

import numpy as np
import pytest
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarLoader
from backend.preprocessing.config import PreprocessingConfig
from backend.preprocessing.filters import (
    remove_invalid_points,
    filter_range,
    filter_height,
    voxel_downsample,
)
from backend.preprocessing.pipeline import PreprocessingPipeline


class TestPreprocessingFilters:
    """Test suite for individual filtering functions."""

    def test_remove_invalid_points_nan_and_inf(self):
        pts = np.array([
            [1.0, 2.0, 0.0, 0.5],
            [np.nan, 2.0, 0.0, 0.5],
            [1.0, np.inf, 0.0, 0.5],
            [1.0, 2.0, -np.inf, 0.5],
            [1.0, 2.0, 0.0, np.nan],
            [3.0, 4.0, 1.0, 0.8],
        ], dtype=np.float32)

        cleaned = remove_invalid_points(pts)
        assert len(cleaned) == 2
        np.testing.assert_allclose(cleaned[0], [1.0, 2.0, 0.0, 0.5])
        np.testing.assert_allclose(cleaned[1], [3.0, 4.0, 1.0, 0.8])

    def test_remove_invalid_empty(self):
        empty = np.empty((0, 4), dtype=np.float32)
        assert len(remove_invalid_points(empty)) == 0

    def test_filter_range_3d(self):
        pts = np.array([
            [0.1, 0.1, 0.1, 0.5],   # dist ~ 0.17 (< 1.0)
            [3.0, 4.0, 0.0, 0.5],   # dist = 5.0 (within [1.0, 10.0])
            [10.0, 10.0, 10.0, 0.5] # dist ~ 17.32 (> 10.0)
        ], dtype=np.float32)

        filtered = filter_range(pts, min_range=1.0, max_range=10.0, use_2d=False)
        assert len(filtered) == 1
        np.testing.assert_allclose(filtered[0], [3.0, 4.0, 0.0, 0.5])

    def test_filter_range_2d(self):
        pts = np.array([
            [3.0, 4.0, 100.0, 0.5], # 2D dist = 5.0, 3D dist ~ 100.12
            [0.1, 0.1, 0.0, 0.5],   # 2D dist ~ 0.14
        ], dtype=np.float32)

        filtered_2d = filter_range(pts, min_range=1.0, max_range=10.0, use_2d=True)
        assert len(filtered_2d) == 1
        np.testing.assert_allclose(filtered_2d[0], [3.0, 4.0, 100.0, 0.5])

    def test_filter_height(self):
        pts = np.array([
            [0.0, 0.0, -3.0, 0.5], # below -1.0
            [0.0, 0.0, -0.5, 0.5], # within
            [0.0, 0.0, 2.0, 0.5],  # within
            [0.0, 0.0, 5.0, 0.5],  # above 3.0
        ], dtype=np.float32)

        filtered = filter_height(pts, min_z=-1.0, max_z=3.0)
        assert len(filtered) == 2
        np.testing.assert_allclose(filtered[:, 2], [-0.5, 2.0])

    def test_voxel_downsample_aggregation(self):
        # Two points in same 0.5m voxel cell
        pts = np.array([
            [0.1, 0.1, 0.1, 0.2],
            [0.3, 0.3, 0.3, 0.4],
            # One point in distinct voxel
            [1.2, 1.2, 1.2, 0.8],
        ], dtype=np.float32)

        downsampled = voxel_downsample(pts, voxel_size=0.5)
        assert len(downsampled) == 2
        
        # Check centroid for the first voxel
        expected_first = np.array([0.2, 0.2, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_allclose(downsampled[0], expected_first, atol=1e-5)

    def test_voxel_downsample_invalid_size(self):
        pts = np.ones((5, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            voxel_downsample(pts, voxel_size=0.0)
        with pytest.raises(ValueError):
            voxel_downsample(pts, voxel_size=-0.5)


class TestPreprocessingConfig:
    """Test suite for configuration loading."""

    def test_default_config(self):
        cfg = PreprocessingConfig()
        assert cfg.enabled is True
        assert cfg.min_range == 0.5
        assert cfg.max_range == 25.0
        assert cfg.min_z == -2.0
        assert cfg.max_z == 4.0
        assert cfg.voxel_downsample_enabled is False

    def test_from_dict(self):
        data = {
            "preprocessing": {
                "enabled": True,
                "min_range": 1.0,
                "max_range": 30.0,
                "min_z": -1.5,
                "max_z": 3.5,
                "voxel_downsample": {
                    "enabled": True,
                    "voxel_size": 0.2,
                }
            }
        }
        cfg = PreprocessingConfig.from_dict(data)
        assert cfg.min_range == 1.0
        assert cfg.max_range == 30.0
        assert cfg.voxel_downsample_enabled is True
        assert cfg.voxel_size == 0.2

    def test_from_yaml_file(self):
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
        cfg = PreprocessingConfig.from_yaml(config_path)
        assert cfg.enabled is True
        assert cfg.min_range == 0.5
        assert cfg.max_range == 25.0


class TestPreprocessingPipeline:
    """Test suite for full pipeline integration."""

    def test_pipeline_disabled_returns_unmodified(self):
        pts = np.array([[100.0, 100.0, 100.0, 0.5]], dtype=np.float32)
        frame = LidarFrame(points=pts)
        cfg = PreprocessingConfig(enabled=False)
        pipeline = PreprocessingPipeline(config=cfg)

        processed = pipeline.process(frame)
        assert len(processed.points) == 1
        np.testing.assert_allclose(processed.points, pts)

    def test_pipeline_filtering_flow(self):
        pts = np.array([
            [1.0, 1.0, 0.0, 0.5],          # Valid
            [np.nan, 0.0, 0.0, 0.5],       # Invalid
            [50.0, 50.0, 0.0, 0.5],        # Out of range
            [2.0, 2.0, 10.0, 0.5],         # Out of Z range
        ], dtype=np.float32)
        frame = LidarFrame(points=pts, frame_id=42, timestamp=1.23)

        cfg = PreprocessingConfig(
            min_range=0.5,
            max_range=20.0,
            min_z=-1.0,
            max_z=3.0,
            voxel_downsample_enabled=False,
        )
        pipeline = PreprocessingPipeline(config=cfg)
        processed = pipeline.process(frame)

        assert processed.frame_id == 42
        assert processed.timestamp == pytest.approx(1.23)
        assert processed.num_points == 1
        np.testing.assert_allclose(processed.points[0], [1.0, 1.0, 0.0, 0.5])
        
        # Verify metadata tracking
        meta = processed.metadata["preprocessing"]
        assert meta["initial_points"] == 4
        assert meta["final_points"] == 1
        assert meta["total_removed"] == 3
        assert meta["stages"]["invalid_removed"] == 1
        assert meta["stages"]["out_of_range_removed"] == 1
        assert meta["stages"]["out_of_height_removed"] == 1

    def test_pipeline_with_synthetic_frame(self):
        loader = SyntheticLidarLoader(num_frames=1, seed=42)
        raw_frame = loader[0]
        
        cfg = PreprocessingConfig(
            min_range=0.5,
            max_range=25.0,
            min_z=-0.5,
            max_z=4.0,
            voxel_downsample_enabled=True,
            voxel_size=0.1,
        )
        pipeline = PreprocessingPipeline(config=cfg)
        processed = pipeline.process(raw_frame)

        assert isinstance(processed, LidarFrame)
        assert processed.num_points < raw_frame.num_points
        assert processed.num_points > 0
        min_xyz, max_xyz = processed.bounds
        assert min_xyz[2] >= -0.5
        assert max_xyz[2] <= 4.0
