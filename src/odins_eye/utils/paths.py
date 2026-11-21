"""Project-relative paths, so code never depends on the current working directory."""

from pathlib import Path

# src/odins_eye/utils/paths.py -> parents[3] is the repository root (editable install)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
