#!/usr/bin/env python3
"""
Phase 0 Health Check Script
Verifies:
1. Python version (>= 3.10)
2. Project folders exist
3. config.yaml exists and loads
4. outputs folder is writable
"""

import sys
import os
from pathlib import Path
import tempfile
import yaml

REQUIRED_DIRS = [
    "backend/data",
    "backend/preprocessing",
    "backend/perception",
    "backend/terrain",
    "backend/priority",
    "backend/mapping",
    "backend/temporal",
    "backend/budget",
    "backend/metrics",
    "app",
    "configs",
    "data",
    "scripts",
    "tests",
    "outputs",
]

def check_python_version() -> bool:
    print(f"[*] Checking Python version: {sys.version.split()[0]}... ", end="")
    if sys.version_info >= (3, 10):
        print("OK (Python >= 3.10)")
        return True
    print("FAILED (Requires Python >= 3.10)")
    return False

def check_directories(root_path: Path) -> bool:
    print("[*] Checking required project folders...")
    all_ok = True
    for rel_dir in REQUIRED_DIRS:
        d = root_path / rel_dir
        if d.is_dir():
            print(f"    [+] {rel_dir}: OK")
        else:
            print(f"    [-] {rel_dir}: MISSING")
            all_ok = False
    return all_ok

def check_config_yaml(root_path: Path) -> bool:
    config_file = root_path / "configs" / "config.yaml"
    print(f"[*] Checking config file: {config_file}... ", end="")
    if not config_file.is_file():
        print("FAILED (File not found)")
        return False
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "project" in data:
            print("OK (Valid YAML loaded)")
            return True
        print("FAILED (Config content invalid)")
        return False
    except Exception as e:
        print(f"FAILED (YAML parse error: {e})")
        return False

def check_outputs_writable(root_path: Path) -> bool:
    outputs_dir = root_path / "outputs"
    print(f"[*] Checking outputs folder writability: {outputs_dir}... ", end="")
    if not outputs_dir.exists():
        print("FAILED (Outputs directory does not exist)")
        return False
    try:
        test_file = outputs_dir / ".write_test_tmp"
        with open(test_file, "w") as f:
            f.write("write_test")
        test_file.unlink()
        print("OK (Writable)")
        return True
    except Exception as e:
        print(f"FAILED (Not writable: {e})")
        return False

def main() -> int:
    root_path = Path(__file__).resolve().parent.parent
    print("=" * 60)
    print("Phase 0 Project Health Check")
    print(f"Root: {root_path}")
    print("=" * 60)

    checks = [
        check_python_version(),
        check_directories(root_path),
        check_config_yaml(root_path),
        check_outputs_writable(root_path),
    ]

    print("=" * 60)
    if all(checks):
        print("All Phase 0 health checks PASSED successfully.")
        return 0
    else:
        print("Some health checks FAILED. Please review above logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
