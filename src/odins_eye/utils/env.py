"""Runtime facts stored with every run: where and on what hardware it ran."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess


def runtime_info() -> dict:
    import torch

    info = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            ".".join(map(str, torch.cuda.get_device_capability(0)))
            if torch.cuda.is_available()
            else None
        ),
        "mps": torch.backends.mps.is_available(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    if shutil.which("nvidia-smi"):
        q = "--query-gpu=name,driver_version,memory.total"
        out = subprocess.run(
            ["nvidia-smi", q, "--format=csv,noheader"], capture_output=True, text=True
        )
        info["nvidia_smi"] = out.stdout.strip()
    return info
