"""
Unit & Integration Tests for DatasetReplaySource (Phase 17 Part B).

Tests:
    1.  DatasetReplaySource implements BaseLidarSource interface.
    2.  Sequential frame replay from KittiLidarLoader.
    3.  Sequential frame replay from NumpyLidarLoader.
    4.  start_index offset: playback begins at correct position.
    5.  max_frames limit: stops after N frames regardless of dataset size.
    6.  loop=False: source stops at end-of-dataset naturally.
    7.  loop=True: playhead wraps to start_index after last frame.
    8.  loop=True with max_frames: bounded infinite loop.
    9.  Iteration protocol (__iter__) yields correct frame sequence.
    10. Validation pass-through: malformed frames rejected with LidarFrameValidationError.
    11. validate_frames=False: no validation, raw frames returned as-is.
    12. create_replay_source() with .npy file.
    13. create_replay_source() with .bin file.
    14. create_replay_source() with directory of .npy files.
    15. create_replay_source() with directory of .bin files.
    16. DatasetReplaySource rejects non-BaseLidarLoader input.
    17. DatasetReplaySource rejects empty loader.
    18. DatasetReplaySource rejects out-of-range start_index.
    19. frames_yielded and current_index properties.
    20. Pipeline integration: RealTimeLidarPipeline consumes replay frames.
"""

import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import pytest

from backend.data.frame import LidarFrame
from backend.data.base import BaseLidarLoader
from backend.data.kitti_loader import KittiLidarLoader
from backend.data.numpy_loader import NumpyLidarLoader
from backend.ingestion import (
    BaseLidarSource,
    DatasetReplaySource,
    create_replay_source,
    LidarFrameValidationError,
)
from backend.streaming.pipeline import RealTimeLidarPipeline
from backend.streaming.config import RealTimePipelineConfig
from backend.mapping.config import MappingConfig


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_points(n: int = 500, seed: int = 0) -> np.ndarray:
    """Create a minimal valid (N, 4) float32 point cloud."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-20.0, 20.0, (n, 4)).astype(np.float32)
    # Ensure z has some variance (no NaN / Inf)
    pts[:, 2] = rng.uniform(-1.0, 2.0, n).astype(np.float32)
    return pts


class _StubLoader(BaseLidarLoader):
    """In-memory loader for testing — no file I/O required."""

    def __init__(self, num_frames: int = 5, n_points: int = 300):
        self._frames: List[LidarFrame] = [
            LidarFrame(
                points=_make_points(n=n_points, seed=i),
                frame_id=i,
                timestamp=float(i) * 0.1,
            )
            for i in range(num_frames)
        ]

    def __len__(self) -> int:
        return len(self._frames)

    def get_frame(self, index: int) -> LidarFrame:
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Index {index} out of range.")
        return self._frames[index]


def _npy_tmpdir(n_files: int = 4, n_points: int = 300) -> Path:
    """Create a temporary directory with n_files .npy point cloud files."""
    tmp = tempfile.mkdtemp(prefix="replay_npy_")
    tmp_path = Path(tmp)
    for i in range(n_files):
        pts = _make_points(n=n_points, seed=i)
        np.save(tmp_path / f"frame_{i:04d}.npy", pts)
    return tmp_path


def _bin_tmpfile(n_points: int = 300) -> Path:
    """Create a single temporary KITTI .bin file with n_points points."""
    tmp = tempfile.mkdtemp(prefix="replay_bin_")
    tmp_path = Path(tmp)
    pts = _make_points(n=n_points, seed=7).flatten()
    path = tmp_path / "scan_0000.bin"
    pts.tofile(str(path))
    return path


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------

class TestDatasetReplaySource:

    # ------------------------------------------------------------------
    # 1. Interface compliance
    # ------------------------------------------------------------------

    def test_implements_base_lidar_source(self):
        """1. DatasetReplaySource is a BaseLidarSource."""
        loader = _StubLoader(num_frames=3)
        source = DatasetReplaySource(loader=loader)
        assert isinstance(source, BaseLidarSource)

    # ------------------------------------------------------------------
    # 2. KITTI dataset replay
    # ------------------------------------------------------------------

    def test_kitti_loader_replay(self):
        """2. Replay frames from a real KITTI .bin file via KittiLidarLoader."""
        bin_path = _bin_tmpfile(n_points=500)
        loader = KittiLidarLoader([bin_path], frame_rate_hz=10.0)
        assert len(loader) == 1

        source = DatasetReplaySource(loader=loader)
        source.start()
        assert source.is_active() is True

        frame = source.get_next_frame()
        assert frame is not None
        assert isinstance(frame, LidarFrame)
        assert frame.points.shape[1] == 4

        # Next call → exhausted, loop=False
        none_frame = source.get_next_frame()
        assert none_frame is None
        assert source.is_active() is False

        source.stop()

    # ------------------------------------------------------------------
    # 3. NumPy dataset replay
    # ------------------------------------------------------------------

    def test_numpy_loader_replay(self):
        """3. Replay frames from .npy files via NumpyLidarLoader."""
        npy_dir = _npy_tmpdir(n_files=3, n_points=400)
        loader = NumpyLidarLoader(npy_dir, frame_rate_hz=10.0)
        assert len(loader) == 3

        source = DatasetReplaySource(loader=loader)
        frames = list(source)

        assert len(frames) == 3
        for f in frames:
            assert isinstance(f, LidarFrame)
            assert f.points.shape[1] == 4

    # ------------------------------------------------------------------
    # 4. start_index offset
    # ------------------------------------------------------------------

    def test_start_index_offset(self):
        """4. Playback starts at start_index, not necessarily index 0."""
        loader = _StubLoader(num_frames=6)
        source = DatasetReplaySource(loader=loader, start_index=2)
        frames = list(source)

        assert len(frames) == 4  # frames 2, 3, 4, 5
        assert frames[0].frame_id == 2
        assert frames[-1].frame_id == 5

    # ------------------------------------------------------------------
    # 5. max_frames limit
    # ------------------------------------------------------------------

    def test_max_frames_limit(self):
        """5. Source stops after max_frames frames even if more are available."""
        loader = _StubLoader(num_frames=10)
        source = DatasetReplaySource(loader=loader, max_frames=4)
        frames = list(source)

        assert len(frames) == 4

    # ------------------------------------------------------------------
    # 6. loop=False stops at end-of-dataset
    # ------------------------------------------------------------------

    def test_no_loop_stops_at_end(self):
        """6. loop=False: source deactivates after last frame."""
        loader = _StubLoader(num_frames=3)
        source = DatasetReplaySource(loader=loader, loop=False)
        source.start()

        collected = []
        while source.is_active():
            f = source.get_next_frame()
            if f is None:
                break
            collected.append(f)

        assert len(collected) == 3
        assert source.is_active() is False

    # ------------------------------------------------------------------
    # 7. loop=True wraps playhead
    # ------------------------------------------------------------------

    def test_loop_wraps_around(self):
        """7. loop=True: playhead wraps to start_index after dataset end."""
        loader = _StubLoader(num_frames=3)
        source = DatasetReplaySource(loader=loader, loop=True, max_frames=7)
        frames = list(source)

        # Should get 7 frames cycling: 0,1,2, 0,1,2, 0
        assert len(frames) == 7
        assert frames[0].frame_id == 0
        assert frames[2].frame_id == 2
        assert frames[3].frame_id == 0  # Wrapped
        assert frames[6].frame_id == 0

    # ------------------------------------------------------------------
    # 8. loop=True with max_frames bounds the loop
    # ------------------------------------------------------------------

    def test_loop_with_max_frames(self):
        """8. loop=True with max_frames yields exactly max_frames frames."""
        loader = _StubLoader(num_frames=2)
        source = DatasetReplaySource(loader=loader, loop=True, max_frames=10)
        frames = list(source)
        assert len(frames) == 10

    # ------------------------------------------------------------------
    # 9. Iteration protocol
    # ------------------------------------------------------------------

    def test_iter_protocol_autostart_autostop(self):
        """9. __iter__ auto-starts and auto-stops the source."""
        loader = _StubLoader(num_frames=4)
        source = DatasetReplaySource(loader=loader)

        assert source.is_active() is False  # Not started yet
        frames = list(source)
        assert len(frames) == 4
        assert source.is_active() is False  # Auto-stopped after iteration

    # ------------------------------------------------------------------
    # 10. Validation rejects malformed frames
    # ------------------------------------------------------------------

    def test_validation_rejects_nan_frame(self):
        """10. Malformed (NaN) frames raise LidarFrameValidationError when validate_frames=True."""

        class _NaNLoader(BaseLidarLoader):
            def __len__(self):
                return 1

            def get_frame(self, index: int) -> LidarFrame:
                nan_pts = np.full((100, 4), np.nan, dtype=np.float32)
                return LidarFrame(points=nan_pts, frame_id=index, timestamp=float(index))

        source = DatasetReplaySource(loader=_NaNLoader(), validate_frames=True)
        source.start()

        with pytest.raises(LidarFrameValidationError, match="non-finite values"):
            source.get_next_frame()

    # ------------------------------------------------------------------
    # 11. validate_frames=False skips validation
    # ------------------------------------------------------------------

    def test_no_validation_passes_raw_frame(self):
        """11. validate_frames=False: raw frame returned without validation check."""

        class _RawLoader(BaseLidarLoader):
            def __len__(self):
                return 1

            def get_frame(self, index: int) -> LidarFrame:
                # Intentionally empty (would fail validation if enabled)
                pts = np.empty((0, 4), dtype=np.float32)
                return LidarFrame(points=pts, frame_id=index, timestamp=0.0)

        source = DatasetReplaySource(loader=_RawLoader(), validate_frames=False)
        source.start()
        frame = source.get_next_frame()
        assert frame is not None
        assert frame.points.shape[0] == 0  # Empty, but not rejected

    # ------------------------------------------------------------------
    # 12. create_replay_source() with .npy file
    # ------------------------------------------------------------------

    def test_create_replay_source_npy_file(self):
        """12. create_replay_source() auto-detects and wraps a single .npy file."""
        npy_dir = _npy_tmpdir(n_files=2, n_points=300)
        npy_file = sorted(npy_dir.glob("*.npy"))[0]

        source = create_replay_source(npy_file, frame_rate_hz=10.0)
        assert isinstance(source, DatasetReplaySource)
        assert source.total_frames == 1

        frames = list(source)
        assert len(frames) == 1

    # ------------------------------------------------------------------
    # 13. create_replay_source() with .bin file
    # ------------------------------------------------------------------

    def test_create_replay_source_bin_file(self):
        """13. create_replay_source() auto-detects and wraps a single .bin file."""
        bin_path = _bin_tmpfile(n_points=400)

        source = create_replay_source(bin_path, frame_rate_hz=10.0)
        assert isinstance(source, DatasetReplaySource)
        assert source.total_frames == 1

        frames = list(source)
        assert len(frames) == 1
        assert frames[0].points.shape == (400, 4)

    # ------------------------------------------------------------------
    # 14. create_replay_source() with directory of .npy files
    # ------------------------------------------------------------------

    def test_create_replay_source_npy_directory(self):
        """14. create_replay_source() handles directory of .npy files."""
        npy_dir = _npy_tmpdir(n_files=5, n_points=200)

        source = create_replay_source(npy_dir, format_hint="numpy", frame_rate_hz=5.0)
        assert isinstance(source, DatasetReplaySource)
        assert source.total_frames == 5

        frames = list(source)
        assert len(frames) == 5

    # ------------------------------------------------------------------
    # 15. create_replay_source() with directory of .bin files
    # ------------------------------------------------------------------

    def test_create_replay_source_bin_directory(self):
        """15. create_replay_source() handles directory of .bin files."""
        tmp = tempfile.mkdtemp(prefix="replay_bin_dir_")
        tmp_path = Path(tmp)
        for i in range(3):
            pts = _make_points(n=250, seed=i).flatten()
            (tmp_path / f"scan_{i:04d}.bin").write_bytes(pts.tobytes())

        source = create_replay_source(tmp_path, format_hint="kitti", frame_rate_hz=10.0)
        assert isinstance(source, DatasetReplaySource)
        assert source.total_frames == 3

        frames = list(source)
        assert len(frames) == 3

    # ------------------------------------------------------------------
    # 16. Rejects non-BaseLidarLoader input
    # ------------------------------------------------------------------

    def test_rejects_non_loader(self):
        """16. DatasetReplaySource raises TypeError for non-BaseLidarLoader input."""
        with pytest.raises(TypeError, match="BaseLidarLoader"):
            DatasetReplaySource(loader="not_a_loader")  # type: ignore

    # ------------------------------------------------------------------
    # 17. Rejects empty loader
    # ------------------------------------------------------------------

    def test_rejects_empty_loader(self):
        """17. DatasetReplaySource raises ValueError for empty loader."""

        class _EmptyLoader(BaseLidarLoader):
            def __len__(self):
                return 0

            def get_frame(self, index: int) -> LidarFrame:
                raise IndexError("empty")

        with pytest.raises(ValueError, match="empty loader"):
            DatasetReplaySource(loader=_EmptyLoader())

    # ------------------------------------------------------------------
    # 18. Rejects out-of-range start_index
    # ------------------------------------------------------------------

    def test_rejects_invalid_start_index(self):
        """18. DatasetReplaySource raises IndexError for start_index out of bounds."""
        loader = _StubLoader(num_frames=3)

        with pytest.raises(IndexError, match="out of range"):
            DatasetReplaySource(loader=loader, start_index=10)

        with pytest.raises(IndexError, match="out of range"):
            DatasetReplaySource(loader=loader, start_index=-1)

    # ------------------------------------------------------------------
    # 19. frames_yielded and current_index properties
    # ------------------------------------------------------------------

    def test_frames_yielded_and_current_index(self):
        """19. frames_yielded and current_index advance correctly."""
        loader = _StubLoader(num_frames=5)
        source = DatasetReplaySource(loader=loader)
        source.start()

        assert source.frames_yielded == 0
        assert source.current_index == 0

        for step in range(1, 4):
            source.get_next_frame()
            assert source.frames_yielded == step
            assert source.current_index == step

        source.stop()

    # ------------------------------------------------------------------
    # 20. Pipeline integration
    # ------------------------------------------------------------------

    def test_pipeline_consumes_replay_frames(self):
        """20. RealTimeLidarPipeline successfully processes frames from DatasetReplaySource."""
        map_cfg = MappingConfig(
            resolution=0.2,
            min_x=-25.0,
            max_x=25.0,
            min_y=-25.0,
            max_y=25.0,
        )
        stream_cfg = RealTimePipelineConfig(
            max_buffer_size=10,
            collect_timing_metrics=True,
        )
        pipeline = RealTimeLidarPipeline(config=stream_cfg, mapping_config=map_cfg)

        loader = _StubLoader(num_frames=4, n_points=2000)
        source = DatasetReplaySource(loader=loader, validate_frames=True)

        results = []
        for frame in source:
            res = pipeline.process_frame(frame)
            results.append(res)

        assert len(results) == 4
        assert results[0].is_initial_frame is True
        assert all(r.success for r in results)
        assert all(r.adaptive_map is not None for r in results)
