"""
Unit tests for LiDAR Data Loader Foundation (Phase 1).
"""

import numpy as np
import pytest

from backend.data.frame import LidarFrame
from backend.data.synthetic import SyntheticLidarGenerator, SyntheticLidarLoader


class TestLidarFrame:
    """Test suite for LidarFrame data structure."""

    def test_valid_frame_creation(self):
        pts = np.array([
            [1.0, 2.0, 0.5, 0.8],
            [3.0, 4.0, 1.2, 0.4],
            [-2.0, -1.0, -0.1, 0.2],
        ], dtype=np.float32)

        frame = LidarFrame(points=pts, frame_id=1, timestamp=0.1)

        assert frame.num_points == 3
        assert frame.points.shape == (3, 4)
        assert frame.xyz.shape == (3, 3)
        assert frame.intensity.shape == (3,)
        np.testing.assert_allclose(frame.xyz[0], [1.0, 2.0, 0.5])
        assert frame.intensity[0] == pytest.approx(0.8)

    def test_bounds_calculation(self):
        pts = np.array([
            [-5.0, 10.0, -1.0, 0.5],
            [15.0, -20.0, 3.5, 0.9],
        ], dtype=np.float32)

        frame = LidarFrame(points=pts)
        min_xyz, max_xyz = frame.bounds

        np.testing.assert_allclose(min_xyz, [-5.0, -20.0, -1.0])
        np.testing.assert_allclose(max_xyz, [15.0, 10.0, 3.5])

    def test_empty_frame_bounds(self):
        pts = np.empty((0, 4), dtype=np.float32)
        frame = LidarFrame(points=pts)
        min_xyz, max_xyz = frame.bounds
        assert frame.num_points == 0
        np.testing.assert_allclose(min_xyz, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(max_xyz, [0.0, 0.0, 0.0])

    def test_invalid_shapes_raise_value_error(self):
        # 1D array
        with pytest.raises(ValueError):
            LidarFrame(points=np.array([1.0, 2.0, 3.0, 4.0]))

        # 3 columns (missing intensity)
        with pytest.raises(ValueError):
            LidarFrame(points=np.ones((10, 3)))

        # 5 columns
        with pytest.raises(ValueError):
            LidarFrame(points=np.ones((10, 5)))

    def test_get_summary(self):
        pts = np.array([
            [0.0, 0.0, 0.0, 0.1],
            [1.0, 1.0, 1.0, 0.9],
        ], dtype=np.float32)
        frame = LidarFrame(points=pts, frame_id="sample_001")
        summary = frame.get_summary()

        assert summary["frame_id"] == "sample_001"
        assert summary["num_points"] == 2
        assert summary["shape"] == (2, 4)
        assert summary["intensity_range"] == [pytest.approx(0.1), pytest.approx(0.9)]


class TestSyntheticLidarGenerator:
    """Test suite for synthetic LiDAR frame generator."""

    def test_ground_generation(self):
        gen = SyntheticLidarGenerator(seed=123)
        ground = gen.generate_ground(num_points=1000, radius_max=20.0, ground_z=0.0)

        assert ground.shape == (1000, 4)
        # Ground z should be close to 0 with low variance
        assert np.mean(ground[:, 2]) == pytest.approx(0.0, abs=0.05)
        # Ground intensity should be in 0.10 - 0.35
        assert np.all(ground[:, 3] >= 0.09)
        assert np.all(ground[:, 3] <= 0.36)

    def test_box_obstacle_generation(self):
        gen = SyntheticLidarGenerator(seed=123)
        box = gen.generate_box_obstacle(center_x=5.0, center_y=2.0, height=2.0, num_points=200)

        assert box.shape[1] == 4
        # Points should have z between 0 and 2.0
        assert np.all(box[:, 2] >= 0.0)
        assert np.all(box[:, 2] <= 2.05)
        # Points should be centered around (5.0, 2.0)
        assert np.mean(box[:, 0]) == pytest.approx(5.0, abs=1.5)
        assert np.mean(box[:, 1]) == pytest.approx(2.0, abs=1.5)

    def test_cylinder_obstacle_generation(self):
        gen = SyntheticLidarGenerator(seed=123)
        cylinder = gen.generate_cylinder_obstacle(center_x=4.0, center_y=-3.0, radius=0.5, height=3.0, num_points=100)

        assert cylinder.shape == (100, 4)
        assert np.all(cylinder[:, 2] >= 0.0)
        assert np.all(cylinder[:, 2] <= 3.0)

    def test_generate_frame_integration(self):
        gen = SyntheticLidarGenerator(seed=42)
        frame = gen.generate_frame(frame_id=0, num_ground_points=4000)

        assert isinstance(frame, LidarFrame)
        assert frame.num_points > 4000
        assert frame.points.shape[1] == 4
        assert "ground_points_count" in frame.metadata
        assert "obstacle_points_count" in frame.metadata


class TestSyntheticLidarLoader:
    """Test suite for SyntheticLidarLoader."""

    def test_loader_length_and_indexing(self):
        loader = SyntheticLidarLoader(num_frames=4, frame_rate_hz=10.0, seed=42)
        assert len(loader) == 4

        frame0 = loader[0]
        assert frame0.frame_id == 0
        assert frame0.timestamp == pytest.approx(0.0)

        frame1 = loader[1]
        assert frame1.frame_id == 1
        assert frame1.timestamp == pytest.approx(0.1)

    def test_loader_index_out_of_bounds(self):
        loader = SyntheticLidarLoader(num_frames=2)
        with pytest.raises(IndexError):
            _ = loader[2]
        with pytest.raises(IndexError):
            _ = loader[-1]

    def test_loader_iteration(self):
        loader = SyntheticLidarLoader(num_frames=3, seed=42)
        frames = list(loader)
        assert len(frames) == 3
        for idx, f in enumerate(frames):
            assert f.frame_id == idx

    def test_seed_reproducibility(self):
        loader1 = SyntheticLidarLoader(num_frames=1, seed=999)
        loader2 = SyntheticLidarLoader(num_frames=1, seed=999)

        np.testing.assert_allclose(loader1[0].points, loader2[0].points)
