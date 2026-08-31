"""
Base Phase 0 Test Suite
Verifies directory structure, configuration integrity, and environment sanity.
"""

import os
import sys
from pathlib import Path
import pytest
import yaml

from run_demo import load_config

ROOT_DIR = Path(__file__).resolve().parent.parent

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


def test_python_version():
    """Verify Python version is 3.10+ (specifically Python 3.12 in environment)."""
    assert sys.version_info >= (3, 10), f"Unsupported Python version: {sys.version}"


@pytest.mark.parametrize("rel_dir", REQUIRED_DIRS)
def test_required_directory_exists(rel_dir):
    """Verify each required project directory exists."""
    target_dir = ROOT_DIR / rel_dir
    assert target_dir.exists(), f"Required directory '{rel_dir}' does not exist."
    assert target_dir.is_dir(), f"Path '{rel_dir}' is not a directory."


def test_config_yaml_validity():
    """Verify config.yaml exists and contains required keys."""
    config_file = ROOT_DIR / "configs" / "config.yaml"
    assert config_file.is_file(), "configs/config.yaml does not exist."
    
    config = load_config(config_file)
    assert isinstance(config, dict), "Config must parse into a dictionary."
    assert "project" in config, "Config is missing 'project' section."
    assert "name" in config["project"], "Config missing 'project.name'."
    assert config["project"]["name"] == "adaptive-lidar"


def test_outputs_directory_writable(tmp_path):
    """Verify the outputs folder is writable."""
    outputs_dir = ROOT_DIR / "outputs"
    assert outputs_dir.is_dir(), "outputs directory does not exist."
    
    test_file = outputs_dir / ".pytest_write_probe"
    try:
        test_file.write_text("ok", encoding="utf-8")
        assert test_file.read_text(encoding="utf-8") == "ok"
    finally:
        if test_file.exists():
            test_file.unlink()
