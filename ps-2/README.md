# Adaptive Variable-Resolution 2.5D LiDAR Mapping

An intelligent, resource-efficient 2.5D elevation and occupancy mapping framework dynamically allocating spatial resolution based on terrain complexity, perception priorities, and budget constraints.

## Project Structure

```
.
├── app/                  # Application and dashboard layer
├── backend/              # Core processing and mapping modules
│   ├── budget/           # Resource & computation budget management
│   ├── data/             # Data loading and sensor interfaces
│   ├── mapping/          # Variable-resolution 2.5D grid mapping
│   ├── metrics/          # Evaluation metrics and benchmarks
│   ├── perception/       # Object detection / obstacle perception
│   ├── preprocessing/    # Ground filtering and outlier removal
│   ├── priority/         # Priority map generation
│   ├── temporal/         # Multi-frame temporal integration
│   └── terrain/          # Terrain slope/roughness analysis
├── configs/              # System configuration files
│   └── config.yaml       # Primary system configuration
├── data/                 # Raw/processed LiDAR datasets
├── outputs/              # Generated maps, logs, and artifacts
├── scripts/              # Helper and diagnostic scripts
│   └── health_check.py   # Environment and structure health check
├── tests/                # Test suite (pytest)
│   └── test_basic.py     # Base environment & configuration tests
├── requirements.txt      # Python dependencies
└── run_demo.py           # Demo runner entrypoint
```

## Setup & Health Check

1. Activate virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run health check:
   ```bash
   python scripts/health_check.py
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Run demo runner:
   ```bash
   python run_demo.py
   ```
