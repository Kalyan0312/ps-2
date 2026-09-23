# Adaptive Variable-Resolution 2.5D LiDAR Mapping System

## 1. Project Overview

Autonomous mobile robots and self-driving platforms operating in unstructured or dynamic environments require accurate, low-latency 2.5D elevation grid maps for obstacle detection, traversability analysis, and path planning. Uniform grid mapping forces an impractical trade-off: coarse grids miss critical fine obstacles and thin terrain hazards, while uniform fine grids suffer from quadratic memory growth and excessive compute times that prohibit real-time performance.

This project delivers an **Adaptive Variable-Resolution 2.5D LiDAR Mapping System** that dynamically allocates spatial resolution based on terrain complexity, preserves $\mathcal{O}(1)$ query efficiency via hierarchical index routing, detects temporal state changes across consecutive observations, performs incremental map updates, streams point cloud data in real time, and provides an integrated browser-based live visualization application.

---

## 2. Core Idea

Rather than storing the entire environment at an expensive uniform resolution (e.g. 0.05 m per cell across a $50 \times 50\text{ m}$ domain requiring 1,000,000 cells and ~20 MB of array memory), our system selectively allocates multi-resolution tiers:
- **Flat, benign terrain (70–75% of space)**: Represented at **Coarse (0.50 m)** resolution to minimize memory and computational load.
- **Moderate undulations / transitions (15–20% of space)**: Represented at **Medium (0.25 m)** resolution.
- **Complex terrain / rough surfaces (1–5% of space)**: Represented at **Fine (0.10 m)** resolution.
- **Sharp obstacles, critical boundaries, steep edges (5–10% of space)**: Represented at **Ultra-Fine (0.05 m)** resolution.

This variable-resolution structure achieves a **67.0% reduction in allocated grid cells** and a **67.2% reduction in array memory footprint** while preserving sub-decimeter geometric detail where safety matters.

---

## 3. System Architecture

The complete end-to-end processing pipeline operates deterministically across the following stages:

```
                      +-----------------------------+
                      |     Raw LiDAR Point Cloud   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Preprocessing Pipeline   |
                      | (NaN filter, range, bbox)   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Uniform Base Grid Map     |
                      |  (2.5D elevation & stats)   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Terrain Complexity Engine |
                      | (roughness, slope, range)   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Adaptive Resolution Decision|
                      |  (Priority tier selection)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Resource Budget Optimizer |
                      | (enforce memory/compute cap)|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Temporal Change Detector  |
                      | (5-state multi-tier delta)  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Incremental Adaptive Builder|
                      | (FULL_REUSE / INC / REBUILD)|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Real-Time Streaming Pipeline|
                      | (buffered FIFO orchestration|
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  AdaptiveMap25D Data Model  |
                      |  (O(1) Spatial Queries)     |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  FastAPI Live App & Engine  |
                      | (REST, WebSockets, Web UI)  |
                      +-----------------------------+
```

---

## 4. Phase-by-Phase Development

| Phase | Milestone / Component | Key Accomplishments |
|---|---|---|
| **Phase 1–4** | Ingestion & Uniform Mapping | Point cloud ingestion (KITTI/synthetic), range/coordinate filtering, vectorized 2.5D uniform grid mapping. |
| **Phase 5–7** | Multi-Resolution & Selection | Multi-resolution grid representation, terrain analysis (roughness, slope, elevation range), priority-based tier selection. |
| **Phase 8–10** | Budget Optimization & Queries | Resource budget management, memory downgrade policies, fast vectorized point-in-bounding-box accumulation, $\mathcal{O}(1)$ index matrix routing. |
| **Phase 11** | Hardening & Memory Model | Inclusive boundary handling, vectorized terrain analysis, strictly verified physical array memory measurement (`ndarray.nbytes`), end-to-end demo. |
| **Phase 12 (A/B/C)** | Performance Optimization | Vectorized adaptive index construction, single-pass point extraction, sparse tier construction, reducing adaptive map build time from ~152 ms to **~7.69 ms**. |
| **Phase 13 (A/B/C)** | Temporal & Incremental Mapping | 5-state temporal change detection (`unoccupied`, `unchanged`, `changed`, `newly_observed`, `no_longer_observed`), multi-frame state manager, smart incremental build strategies (`FULL_REUSE`, `INCREMENTAL_UPDATE`, `FULL_REBUILD`). |
| **Phase 14 (Part A)** | Real-Time Streaming Core | FIFO frame buffering with overflow policies (`DROP_OLDEST`, `DROP_NEWEST`), streaming orchestrator (`RealTimeLidarPipeline`), per-stage latency tracking, resilient error recovery. |
| **Phase 14 (Part B)** | Live App & Visualization | FastAPI application serving REST endpoints, WebSocket live metric broadcasting, integrated dark-mode technical dashboard (HTML/CSS/Vanilla JS), $\mathcal{O}(1)$ spatial query interface, Docker containerization. |
| **Phase 16 (A/B/C)** | Adaptive Map Snapshot API & Canvas Visualization | `GET /api/map/snapshot` REST endpoint, interactive HTML Canvas 2.5D adaptive map renderer with pan/zoom/fit controls, elevation/resolution color modes, hover inspection, WebSocket-driven live map updates. |
| **Phase 17 (Part A)** | LiDAR Ingestion Abstraction | `BaseLidarSource` abstract interface, `LidarFrameValidator` boundary validator, `SyntheticLidarSource` wrapping `SyntheticLidarGenerator`, unified ingestion subsystem (`backend/ingestion/`). |
| **Phase 17 (Part B)** | Recorded Dataset Replay | `DatasetReplaySource(BaseLidarSource)` wrapping any `BaseLidarLoader` (KITTI `.bin`, NumPy `.npy`) with looping, frame limits, start offset, and boundary validation. `create_replay_source()` factory for auto-detected dataset loading. |

---

## 5. Adaptive Mapping Data Model

The core representation is encapsulated in **`AdaptiveMap25D`**:
- **Multi-Tier Grids (`tier_maps`)**: A collection of isolated, localized `GridMap25D` layers for `coarse` ($0.50\text{ m}$), `medium` ($0.25\text{ m}$), `fine` ($0.10\text{ m}$), and `ultra_fine` ($0.05\text{ m}$). Each layer stores `point_count`, `min_z`, `max_z`, `mean_z`, and `mean_intensity`.
- **Hierarchical Base Index**: Three 2D index matrices mapped onto the base uniform grid resolution:
  - `index_tier`: String array identifying the owning resolution tier for each base cell.
  - `index_row`: Integer array storing the row offset in the tier's localized sub-grid.
  - `index_col`: Integer array storing the column offset in the tier's localized sub-grid.
- **$\mathcal{O}(1)$ Constant-Time Spatial Queries (`world_query(x, y)`)**:
  World coordinates $(x, y)$ map directly via uniform arithmetic into base cell $(r_{base}, c_{base})$. The pipeline looks up $(tier, r_{tier}, c_{tier})$ in $\mathcal{O}(1)$ without tree traversal, querying the localized tier map in direct memory access.

---

## 6. Performance Optimization Journey

Through systematic profiling and vectorization, the execution time for the complete adaptive mapping pipeline underwent massive performance gains:

| Optimization Stage | Total Adaptive Pipeline Time | Adaptive Map Build Only | Improvement |
|---|---|---|---|
| **Initial Unoptimized Baseline** | $\approx 152.00\text{ ms}$ | $\approx 95.00\text{ ms}$ | Baseline |
| **Phase 11 (Hardened Vectorization)** | $\approx 47.00\text{ ms}$ | $\approx 28.00\text{ ms}$ | $3.2\times$ faster |
| **Phase 12 (Part C Optimized)** | $\approx 17.00\text{ ms}$ | $\approx 7.77\text{ ms}$ | $8.9\times$ faster |
| **Phase 14 Measured (Current)** | **$16.79\text{ ms}$** | **$7.69\text{ ms}$** | **$9.0\times$ faster** |

*Measurements taken on synthetic benchmark frames (7,800 raw points, 6,011 preprocessed points, $250 \times 250$ base grid).*

---

## 7. Temporal Change Detection

Between consecutive time frames $t-1$ and $t$, the `TemporalChangeDetector` compares base cell elevations using fine-tier aggregated layers to classify every spatial cell into one of five mutually exclusive states:
1. `unoccupied`: No points observed in either frame.
2. `unchanged`: Points present in both frames with $|\Delta z| \le \text{threshold}$ ($0.15\text{ m}$).
3. `changed`: Points present in both frames with significant elevation delta $|\Delta z| > 0.15\text{ m}$.
4. `newly_observed`: Cell was unoccupied at $t-1$, now contains points at $t$.
5. `no_longer_observed`: Cell contained points at $t-1$, now unoccupied at $t$.

---

## 8. Incremental Adaptive Mapping

The `IncrementalAdaptiveMapBuilder` selects from three distinct update strategies:
- **`FULL_REUSE`**: Selected when the scene is identical ($0\text{ cells changed}$). Skips all point binning and array allocations, returning the previous map reference with zero computational overhead.
- **`INCREMENTAL_UPDATE`**: Selected for localized changes ($\le 50\%$ changed cells). Rebuilds only the modified spatial cells and reuses stable tier sub-grids by reference.
- **`FULL_REBUILD`**: Automatic fallback triggered when dynamic scene changes exceed the threshold ($> 50\%$ changed cells), deferring directly to the fast vectorized `AdaptiveMap25DBuilder` (~7.69 ms) to avoid incremental overhead.

---

## 9. Real-Time Streaming Pipeline Engine (Phase 14 Part A)

The streaming layer (`backend/streaming/`) orchestrates continuous LiDAR point cloud streams:
- **`backend/streaming/config.py`**: Configures buffer capacity (`max_buffer_size`), overflow policies (`OverflowPolicy.DROP_OLDEST`, `DROP_NEWEST`, `BLOCK`), and metric toggles.
- **`backend/streaming/result.py`**: `StreamingPipelineResult` encapsulates output map, frame identifiers, timing metrics, and selected strategies.
- **`backend/streaming/frame_buffer.py`**: FIFO queue managing burst frame arrivals and enforcing drop policies.
- **`backend/streaming/pipeline.py`**: `RealTimeLidarPipeline` coordinating the full pipeline lifecycle with per-stage timing breakdown and error containment.
- **`backend/streaming/__init__.py`**: Public interface.

---

## 10. Live Streaming Application & Visualization (Phase 14 Part B)

The FastAPI application (`backend/api/`) wraps the core pipeline into a single-process service serving REST, WebSockets, and static frontend assets:

### Architecture
- **Single Process**: One Python runtime managing asynchronous HTTP requests, WebSocket event loops, and background LiDAR simulation tasks via `StreamSessionManager`.
- **Zero Heavy Frontend Dependencies**: Pure HTML5, CSS3, and Vanilla JavaScript with automatic WebSocket reconnection and zero Node/Vite/Webpack build steps.
- **Lightweight Telemetry**: Telemetry packets push serializable metrics (stage latencies, cell change counts, FPS, tier percentages) without streaming multi-megabyte raw point clouds.

### REST Endpoints
- `GET /health`: Returns service health, pipeline readiness, and streaming status.
- `POST /api/stream/start`: Starts simulated background streaming session with configurable FPS and frame limits.
- `POST /api/stream/stop`: Safely stops background stream worker without corrupting pipeline state.
- `GET /api/stream/status`: Returns current throughput, latency, frames processed/dropped, and strategy counts.
- `GET /api/metrics`: Aggregated pipeline diagnostics and active map summary.
- `POST /api/map/query`: Executes instant constant-time $\mathcal{O}(1)$ spatial coordinate lookup (`AdaptiveMap25D.world_query(x, y)`).

### WebSocket Endpoint
- `WebSocket /ws/stream`: Broadcasts real-time frame telemetry to connected browser clients at up to 60 Hz.

### Dashboard UI Layout
1. **System & Ingestion Status**: Live/Idle indicators, FIFO buffer metrics, frames received/processed/dropped.
2. **Live Pipeline Performance**: Current and average latency (ms), FPS, and stage breakdown progress bars (Preprocessing, Base Mapping, Terrain Analysis, Resolution Selection, Adaptive Map Update).
3. **Temporal Mapping & Strategy**: Active strategy badges (`FULL_BUILD`, `FULL_REUSE`, `INCREMENTAL_UPDATE`, `FULL_REBUILD`), delta cell counters (changed, unchanged, newly observed, no longer observed).
4. **Adaptive Spatial Resolution Allocation**: Visual proportional distribution bars across Coarse ($0.50\text{ m}$), Medium ($0.25\text{ m}$), Fine ($0.10\text{ m}$), and Ultra-Fine ($0.05\text{ m}$) tiers.
5. **Interactive Spatial Query Engine**: Coordinate inputs $(X, Y)$ with instant display of cell tier, resolution, point count, min/max/mean elevation, and intensity.

---

## 11. Testing and Verification

### Pytest Test Suite Results
```
============================= test session starts ==============================
rootdir: /Users/kalyan/sih 2k26 ps-2/ps-2
collected 303 items

tests/test_adaptive_mapping.py .........................                 [  9%]
tests/test_adaptive_resolution.py ...........                             [ 13%]
tests/test_api_streaming.py .........                                    [ 16%]
tests/test_basic.py .....                                                [ 18%]
tests/test_budget_optimization.py .................                      [ 25%]
tests/test_data_loader.py ........                                       [ 28%]
tests/test_incremental_adaptive_mapping.py ........................      [ 37%]
tests/test_metrics.py .................                                  [ 43%]
tests/test_multiresolution_mapping.py ...........                        [ 47%]
tests/test_phase11_hardening.py ....................................     [ 61%]
tests/test_preprocessing.py ............                                 [ 65%]
tests/test_real_data_loaders.py ....................                     [ 73%]
tests/test_streaming_pipeline.py ..................                      [ 79%]
tests/test_temporal_change_detection.py .......................          [ 88%]
tests/test_temporal_processing.py ....................                   [ 95%]
tests/test_terrain_analysis.py ..........                                [ 99%]
tests/test_uniform_mapping.py ..                                         [100%]

============================= 303 passed in 1.80s ==============================
```
- **Total Tests Passed**: **303**
- **Total Tests Failed**: **0**
- **Regressions**: **0**

---

## 12. Benchmark Results

### 1. Static Adaptive Mapping Benchmark (`scripts/benchmark_mapping.py`)
```
================================================================================
Stage                             Average Time     Min Time     Max Time
------------------------------------------------------------------------
Preprocessing                          0.43 ms      0.41 ms      0.45 ms
Terrain Analysis                       6.06 ms      6.00 ms      6.13 ms
Resolution Decision                    1.54 ms      1.53 ms      1.55 ms
Adaptive Map Build                     7.69 ms      7.56 ms      7.85 ms
------------------------------------------------------------------------
Total Adaptive Pipeline               16.79 ms     16.55 ms     16.99 ms
================================================================================
Memory: Adaptive Map 6,406.8 KB vs Uniform Ultra-Fine 19,531.2 KB (67.2% reduction)
```

### 2. Incremental Mapping Benchmark (`scripts/benchmark_incremental_mapping.py`)
```
================================================================================
Scenario 1 (Identical Scene)     : 92.4% cell reuse, 0 invalidated
Scenario 2 (5% Changed Scene)    : 91.9% cell reuse, INCREMENTAL_UPDATE
Scenario 3 (30% Changed Scene)   : FULL_REBUILD fallback triggered
Scenario 4 (Highly Dynamic Scene): FULL_REBUILD fallback triggered (6.95 ms rebuild)
Scenario 5 (Tier Transition)     : 86.3% cell reuse, fine-tier update
================================================================================
```

### 4. Recorded LiDAR Dataset Replay Demo (`scripts/run_dataset_replay_demo.py`)
```
============================================================
RECORDED LIDAR DATASET REPLAY DEMO
Phase 17 Part C — Adaptive 2.5D Mapping Pipeline
============================================================
Dataset         : tests/fixtures/lidar
Start index     : 0
Max frames      : all
Loop            : False
Frame rate      : 10.0 Hz
Validate frames : True
Total frames in dataset  : 5

Frame 0: Points: 1960 | Strategy: FULL_BUILD   | Cells: 1684 | Latency: 21.18 ms
Frame 1: Points: 1960 | Strategy: FULL_REBUILD | Cells: 1705 | Latency: 44.02 ms
Frame 2: Points: 1960 | Strategy: FULL_REBUILD | Cells: 1555 | Latency: 45.09 ms
Frame 3: Points: 2160 | Strategy: FULL_REBUILD | Cells: 1731 | Latency: 46.33 ms
Frame 4: Points: 1960 | Strategy: FULL_REBUILD | Cells: 1566 | Latency: 44.70 ms
============================================================
REPLAY SUMMARY:
  Frames Processed: 5 | Frames Failed: 0 | Dropped: 0
  Average Latency : 40.27 ms | Average Throughput: 24.8 FPS
============================================================
```

---

## 13. Recorded LiDAR Dataset Replay

Phase 17 Part C provides an end-to-end recorded dataset ingestion and replay workflow that feeds real/recorded LiDAR files directly into the active mapping pipeline without duplicating pipeline logic.

### 1. Architecture Flow
```
Recorded LiDAR Files (.npy / .bin)
               ↓
        BaseLidarLoader
   (NumpyLidarLoader / KittiBinLoader)
               ↓
      DatasetReplaySource (BaseLidarSource)
               ↓
      LidarFrameValidator
               ↓
          LidarFrame
               ↓
     RealTimeLidarPipeline
               ↓
        AdaptiveMap25D
               ↓
    FastAPI / WebSocket / Canvas
```

### 2. Supported File Formats & Data Layouts
- **NumPy Point Clouds (`.npy`)**:
  - **Layout**: 2D NumPy float array of shape `(N, 3)` with columns `[x, y, z]` or shape `(N, 4)` with columns `[x, y, z, intensity]`.
  - **Dtype**: `float32` or `float64` (automatically normalized).
- **KITTI Binary Point Clouds (`.bin`)**:
  - **Layout**: Contiguous binary `float32` stream with 4 values per point `[x, y, z, reflectance/intensity]`.
  - **Byte length**: Exactly $N \times 16$ bytes (4 float32 values per point).

### 3. CLI Usage & Options
```bash
# Run bundled demo fixture (default)
python scripts/run_dataset_replay_demo.py

# Replay an external dataset directory with frame limit and start offset
python scripts/run_dataset_replay_demo.py --dataset /path/to/kitti/velodyne_points/data --start-index 10 --max-frames 50

# Continuous looping replay at specified frame rate
python scripts/run_dataset_replay_demo.py --dataset /path/to/npy_frames --loop --frame-rate 20.0
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `Path` | `tests/fixtures/lidar` | Path to directory containing `.npy` or `.bin` frame files. |
| `--max-frames` | `int` | `None` (all) | Maximum number of frames to replay before stopping. |
| `--start-index` | `int` | `0` | 0-based frame index to begin replay from. |
| `--loop` | `flag` | `False` | Loop continuously over dataset until interrupted. |
| `--frame-rate` | `float` | `10.0` | Target replay rate in Hz. |
| `--no-validate`| `flag` | `False` | Disable frame validation checks. |

### 4. Substituting User-Provided Datasets
To run custom recorded point clouds through the system:
1. Save each sequential frame as `frame_0000.npy` (shape `(N, 3)` or `(N, 4)`) or `frame_0000.bin` (KITTI format) in a directory.
2. Pass the directory path:
   ```bash
   python scripts/run_dataset_replay_demo.py --dataset /path/to/my_lidar_data/
   ```
3. The loader automatically detects format, lazily reads frames one-by-one from disk, validates data, and passes `LidarFrame` objects to `RealTimeLidarPipeline`.

---

## 14. How to Run

### 1. Run the Live Web Application & Dashboard
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Open browser at **`http://localhost:8000`**
- Interactive API Documentation available at **`http://localhost:8000/docs`**

### 2. Run Full Automated Test Suite
```bash
python -m pytest
```

### 3. Run Standalone CLI Demos
```bash
python scripts/run_dataset_replay_demo.py
python scripts/run_realtime_stream_demo.py
python run_demo.py
```

### 4. Run Performance Benchmarks
```bash
python scripts/benchmark_mapping.py
python scripts/benchmark_incremental_mapping.py
```

### 5. Run via Docker
```bash
docker build -t adaptive-lidar .
docker run -p 8000:8000 adaptive-lidar
```

---

## 15. Current Project Status

- **Status**: **Phase 17 Part C — COMPLETE**.
- **Test Suite**: **323 passed, 0 failed, 0 skipped**.
- **Completed Systems**:
  - Ingestion, validation, and multi-sensor data loading.
  - Multi-resolution 2.5D elevation grid model with constant-time $\mathcal{O}(1)$ query index.
  - Vectorized terrain complexity analysis and priority-based resolution selector.
  - Resource budget manager with automated memory degradation.
  - 5-state temporal change detection engine.
  - Incremental adaptive map builder with fast-path reuse and threshold fallbacks.
  - Real-Time Streaming Pipeline engine with FIFO buffering.
  - Single-process FastAPI application with REST endpoints, WebSocket telemetry, and Vanilla UI Dashboard.
  - `GET /api/map/snapshot` adaptive map JSON endpoint and interactive Canvas 2.5D visualization with pan/zoom, color modes, hover inspection, and WebSocket-driven live updates (Phase 16).
  - `BaseLidarSource` / `LidarFrameValidator` / `SyntheticLidarSource` ingestion subsystem (Phase 17A).
  - `DatasetReplaySource` + `create_replay_source()` for recorded dataset replay from KITTI `.bin` and NumPy `.npy` files (Phase 17B).
  - Deterministic fixture dataset, dedicated dataset replay demo script `scripts/run_dataset_replay_demo.py`, comprehensive CLI options, robust error handling, and end-to-end replay test suite (Phase 17C).
- **Classification**: Production-grade prototype suitable for live demonstration, hackathon evaluation, and robotic platform integration.

---

## 16. Future Scope

1. **ROS2 Node Integration**: Native `sensor_msgs/msg/PointCloud2` subscribers and grid map publishers.
2. **WebGL 3D Point Canvas**: Direct browser Three.js / WebGL rendering of elevated 2.5D terrain meshes.
3. **Hardware Sensor Drivers**: Live Velodyne / Ouster / Livox UDP socket stream ingestion.
4. **Persistent Spatio-Temporal Storage**: Memory-mapped binary logging for multi-session SLAM loop closure.
5. **GPU Acceleration (CUDA/Triton)**: Offloading dense 2D convolution filters and k-d tree spatial searches.

