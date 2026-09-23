# LiDAR Demo Fixture Dataset

Small, deterministic 3-D point cloud dataset used as the built-in demo fixture
for Phase 17 Part C (Recorded LiDAR Dataset Replay).

## Files

| File            | Points | Description                                      |
|-----------------|--------|--------------------------------------------------|
| `frame_0000.npy`| 1 960  | Initial scene — flat ground + static building    |
| `frame_0001.npy`| 1 960  | Minor sensor-motion shift                        |
| `frame_0002.npy`| 1 960  | Small obstacle appears (triggers INCREMENTAL)    |
| `frame_0003.npy`| 2 160  | Large scene change + new tall structure           |
| `frame_0004.npy`| 1 960  | Stable scene (same layout as frame 3)            |

## Data Format

Each `.npy` file contains a **NumPy array of shape `(N, 4)` and dtype `float32`**.

| Column | Field       | Units  | Range          |
|--------|-------------|--------|----------------|
| 0      | x           | meters | [-15.0, 15.0]  |
| 1      | y           | meters | [-15.0, 15.0]  |
| 2      | z           | meters | [-0.1, 4.0]    |
| 3      | intensity   | [0, 1] | [0.2, 0.95]    |

Loading example:
```python
import numpy as np
pts = np.load("frame_0000.npy")   # shape (N, 4)
x, y, z, intensity = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
```

## Reproducibility

All files are **100 % deterministic**. Re-running `generate_fixtures.py` with
the same Python / NumPy version will always produce bit-identical files.

Seed formula: `base_seed = 42 + frame_idx * 7`

## Regenerating the Fixtures

```bash
python tests/fixtures/lidar/generate_fixtures.py
```

## Using Your Own Dataset

The replay pipeline accepts any directory of `.npy` or `.bin` files:

```bash
# NumPy directory
python scripts/run_dataset_replay_demo.py --dataset /path/to/my_npy_scans/

# KITTI binary directory
python scripts/run_dataset_replay_demo.py --dataset /path/to/kitti/velodyne/

# CLI options
python scripts/run_dataset_replay_demo.py \
    --dataset /path/to/dataset/ \
    --max-frames 50 \
    --start-index 10 \
    --loop
```

### KITTI `.bin` Format

Each `.bin` file stores one Velodyne scan as a flat array of `float32` values.
Every point occupies **16 bytes** (4 × 4 bytes):

```
[x  y  z  reflectance]  float32  float32  float32  float32
```

Loading example:
```python
import numpy as np
pts = np.fromfile("scan_000000.bin", dtype=np.float32).reshape(-1, 4)
```

### NumPy `.npy` Format

Shape must be `(N, 3)` or `(N, 4)`.

- `(N, 3)`: columns `[x, y, z]`; intensity defaults to `0.5`.
- `(N, 4)`: columns `[x, y, z, intensity]`; intensity preserved as-is.
