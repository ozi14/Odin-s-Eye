"""Experiment folders: experiments/YYYYMMDD_<id>_<desc>/ with config and git state."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from odins_eye.utils.paths import EXPERIMENTS_DIR, PROJECT_ROOT


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def new_run_dir(exp_id: str, desc: str, cfg: DictConfig | None = None) -> Path:
    """Create a fresh run folder; a same-day rerun gets a -2, -3, ... suffix."""
    base = EXPERIMENTS_DIR / f"{dt.date.today():%Y%m%d}_{exp_id}_{desc}"
    path, n = base, 2
    while path.exists():
        path, n = base.with_name(f"{base.name}-{n}"), n + 1
    path.mkdir(parents=True)
    (path / "git.json").write_text(json.dumps(git_state(), indent=2) + "\n")
    if cfg is not None:
        (path / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return path
