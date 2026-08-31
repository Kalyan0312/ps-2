"""
Unit and Integration Tests for Phase 8: Real LiDAR Data Loaders (NumPy .npy and KITTI .bin).
"""

import pytest
import numpy as np
from pathlib import Path

from backend.data.frame import LidarFrame
from backend.data.numpy_loader import load_numpy_file, NumpyLidarLoader
from backend.data.kitti_loader import load_kitti_bin, KittiLidarLoader
from backend.data.factory import load_lidar_file, create_lidar_loader
from backend.preprocessing.pipeline import PreprocessingPipeline
from backend.preprocessing.config import PreprocessingConfig
from backend.mapping.config import MappingConfig
from backend.mapping.uniform_builder import Uniform25DMapBuilder
from backend.mapping.adaptive_builder import AdaptiveMap25DBuilder
from backend.terrain.analyzer import TerrainAnalyzer
from backend.terrain.config import TerrainConfig
from backend.priority.selector import AdaptiveResolutionSelector
from backend.priority.config import PriorityConfig
from backend.metrics.map_metrics import calculate_adaptive_map_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_npy_nx3(tmp_path):
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [-1.0, -2.0, -3.0],
    ], dtype=np.float32)
    path = tmp_path / "scan_3d.npy"
    np.save(path, pts)
    return path


@pytest.fixture
def temp_npy_nx4(tmp_path):
    pts = np.array([
        [1.0, 2.0, 3.0, 0.8],
        [4.0, 5.0, 6.0, 0.4],
        [-1.0, -2.0, -3.0, 0.1],
    ], dtype=np.float32)
    path = tmp_path / "scan_4d.npy"
    np.save(path, pts)
    return path


@pytest.fixture
def temp_kitti_bin(tmp_path):
    pts = np.array([
        [10.0, 2.0, -1.5, 0.75],
        [15.0, -3.0, -1.2, 0.50],
        [5.0, 0.5, -1.8, 0.90],
    ], dtype=np.float32)
    path = tmp_path / "000000.bin"
    pts.tofile(path)
    return path


@pytest.fixture
def realistic_kitti_bin(tmp_path):
    rng = np.random.default_rng(42)
    num_pts = 300
    r = np.sqrt(rng.uniform(1.0, 100.0, size=num_pts))
    theta = rng.uniform(-np.pi, np.pi, size=num_pts)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.normal(0.0, 0.2, size=num_pts)
    intensity = rng.uniform(0.1, 0.9, size=num_pts)
    pts = np.column_stack([x, y, z, intensity]).astype(np.float32)
    path = tmp_path / "realistic_kitti.bin"
    pts.tofile(path)
    return path


# ---------------------------------------------------------------------------
# Test NumPy Loader
# ---------------------------------------------------------------------------

class TestNumpyLoader:

    def test_load_valid_nx3(self, temp_npy_nx3):
        frame = load_numpy_file(temp_npy_nx3, default_intensity=0.6)
        assert isinstance(frame, LidarFrame)
        assert frame.num_points == 3
        assert frame.points.shape == (3, 4)
        np.testing.assert_allclose(frame.xyz[0], [1.0, 2.0, 3.0])
        assert frame.intensity[0] == pytest.approx(0.6)
        assert frame.metadata["source"] == "numpy"
        assert frame.metadata["default_intensity_used"] is True
        assert frame.metadata["has_native_intensity"] is False

    def test_load_valid_nx4(self, temp_npy_nx4):
        frame = load_numpy_file(temp_npy_nx4)
        assert frame.num_points == 3
        assert frame.points.shape == (3, 4)
        np.testing.assert_allclose(frame.xyz[0], [1.0, 2.0, 3.0])
        assert frame.intensity[0] == pytest.approx(0.8)
        assert frame.metadata["source"] == "numpy"
        assert frame.metadata["has_native_intensity"] is True
        assert frame.metadata["default_intensity_used"] is False

    def test_invalid_shapes_raise_error(self, tmp_path):
        # 1D array
        p1 = tmp_path / "arr_1d.npy"
        np.save(p1, np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Expected 2D point cloud array"):
            load_numpy_file(p1)

        # 2 columns
        p2 = tmp_path / "arr_2col.npy"
        np.save(p2, np.ones((5, 2)))
        with pytest.raises(ValueError, match="Invalid number of point cloud columns"):
            load_numpy_file(p2)

        # 5 columns
        p3 = tmp_path / "arr_5col.npy"
        np.save(p3, np.ones((5, 5)))
        with pytest.raises(ValueError, match="Invalid number of point cloud columns"):
            load_numpy_file(p3)

    def test_nonexistent_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_numpy_file("nonexistent_path_12345.npy")

    def test_numpy_loader_sequence(self, tmp_path):
        for i in range(3):
            pts = np.ones((10, 4), dtype=np.float32) * i
            np.save(tmp_path / f"frame_{i:03d}.npy", pts)

        loader = NumpyLidarLoader(tmp_path, frame_rate_hz=10.0)
        assert len(loader) == 3

        f0 = loader[0]
        assert f0.num_points == 10
        assert f0.timestamp == pytest.approx(0.0)

        f1 = loader[1]
        assert f1.timestamp == pytest.approx(0.1)

        with pytest.raises(IndexError):
            _ = loader[3]


# ---------------------------------------------------------------------------
# Test KITTI Loader
# ---------------------------------------------------------------------------

class TestKittiLoader:

    def test_load_valid_kitti_bin(self, temp_kitti_bin):
        frame = load_kitti_bin(temp_kitti_bin)
        assert isinstance(frame, LidarFrame)
        assert frame.num_points == 3
        assert frame.points.shape == (3, 4)
        np.testing.assert_allclose(frame.xyz[0], [10.0, 2.0, -1.5])
        assert frame.intensity[0] == pytest.approx(0.75)
        assert frame.metadata["source"] == "kitti_bin"
        assert frame.metadata["sensor_coordinates"] == "kitti_velodyne"
        assert frame.metadata["file_size_bytes"] == 3 * 16

    def test_empty_kitti_bin(self, tmp_path):
        empty_bin = tmp_path / "empty.bin"
        empty_bin.touch()
        frame = load_kitti_bin(empty_bin)
        assert frame.num_points == 0
        assert frame.points.shape == (0, 4)

    def test_invalid_file_size_raises_error(self, tmp_path):
        corrupt_bin = tmp_path / "corrupt.bin"
        # 15 bytes (not a multiple of 16)
        corrupt_bin.write_bytes(b"\x00" * 15)
        with pytest.raises(ValueError, match="is not a multiple of 16 bytes"):
            load_kitti_bin(corrupt_bin)

    def test_nonexistent_kitti_file(self):
        with pytest.raises(FileNotFoundError):
            load_kitti_bin("nonexistent_kitti_123.bin")

    def test_kitti_loader_sequence(self, tmp_path):
        for i in range(4):
            pts = np.ones((5, 4), dtype=np.float32) * (i + 1)
            pts.tofile(tmp_path / f"kitti_{i:04d}.bin")

        loader = KittiLidarLoader(tmp_path, frame_rate_hz=10.0)
        assert len(loader) == 4

        f0 = loader[0]
        assert f0.num_points == 5
        assert f0.timestamp == pytest.approx(0.0)

        f2 = loader[2]
        assert f2.timestamp == pytest.approx(0.2)

        with pytest.raises(IndexError):
            _ = loader[-1]


# ---------------------------------------------------------------------------
# Test Factory / Auto-detection
# ---------------------------------------------------------------------------

class TestLoaderFactory:

    def test_load_lidar_file_auto_detect(self, temp_npy_nx4, temp_kitti_bin):
        f_npy = load_lidar_file(temp_npy_nx4)
        assert f_npy.metadata["source"] == "numpy"

        f_bin = load_lidar_file(temp_kitti_bin)
        assert f_bin.metadata["source"] == "kitti_bin"

    def test_unsupported_format_raises_error(self, tmp_path):
        txt_file = tmp_path / "points.txt"
        txt_file.write_text("1.0 2.0 3.0")
        with pytest.raises(ValueError, match="Unsupported LiDAR file format"):
            load_lidar_file(txt_file)

    def test_create_lidar_loader_factory(self, tmp_path):
        # Create npy directory
        npy_dir = tmp_path / "npy_seq"
        npy_dir.mkdir()
        np.save(npy_dir / "000.npy", np.ones((5, 4)))

        loader_npy = create_lidar_loader(npy_dir)
        assert isinstance(loader_npy, NumpyLidarLoader)
        assert len(loader_npy) == 1

        # Create kitti directory
        kitti_dir = tmp_path / "kitti_seq"
        kitti_dir.mkdir()
        np.ones((5, 4), dtype=np.float32).tofile(kitti_dir / "000.bin")

        loader_kitti = create_lidar_loader(kitti_dir)
        assert isinstance(loader_kitti, KittiLidarLoader)
        assert len(loader_kitti) == 1


# ---------------------------------------------------------------------------
# Test End-to-End Pipeline Integration with Real Data
# ---------------------------------------------------------------------------

class TestRealDataPipelineIntegration:

    def test_kitti_frame_through_adaptive_pipeline(self, realistic_kitti_bin):
        # 1. Load real KITTI format point cloud
        frame = load_kitti_bin(realistic_kitti_bin)
        assert frame.num_points == 300

        # 2. Preprocessing
        prep_pipeline = PreprocessingPipeline(config=PreprocessingConfig())
        prep_frame = prep_pipeline.process(frame)
        assert prep_frame.num_points <= frame.num_points

        # 3. Base Uniform Map
        map_cfg = MappingConfig(
            resolution=0.5,
            min_x=-15.0, max_x=15.0,
            min_y=-15.0, max_y=15.0,
        )
        base_builder = Uniform25DMapBuilder(config=map_cfg)
        base_grid = base_builder.build_map(prep_frame)
        assert base_grid.num_occupied_cells > 0

        # 4. Terrain Analysis
        analyzer = TerrainAnalyzer(config=TerrainConfig())
        terrain_result = analyzer.analyze(base_grid)
        assert terrain_result.num_analyzed_cells > 0

        # 5. Adaptive Decision Selection
        selector = AdaptiveResolutionSelector(config=PriorityConfig())
        decision_result = selector.select_resolution(base_grid, terrain_result)
        assert decision_result.num_decided_cells > 0

        # 6. Adaptive Map Reconstruction
        builder = AdaptiveMap25DBuilder(config=map_cfg)
        adaptive_map = builder.build(prep_frame, decision_result)
        assert adaptive_map.num_represented_cells > 0

        # 7. Metrics evaluation
        metrics = calculate_adaptive_map_metrics(adaptive_map)
        assert metrics.adaptive_breakdown is not None
        assert metrics.memory.array_bytes > 0

        # 8. Query test
        assigned_mask = (adaptive_map.index_tier != "")
        found_point = None
        for pt in prep_frame.points:
            br, bc = base_grid.world_to_grid(pt[0], pt[1])
            if 0 <= br < base_grid.rows and 0 <= bc < base_grid.cols and assigned_mask[br, bc]:
                found_point = pt
                break

        assert found_point is not None
        query_info = adaptive_map.world_query(float(found_point[0]), float(found_point[1]))
        assert query_info.is_represented is True
        assert query_info.tier in ["coarse", "medium", "fine", "ultra_fine"]
        assert query_info.mean_z is not None
