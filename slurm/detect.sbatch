#!/bin/bash
#SBATCH --job-name=oe-detect
#SBATCH --partition=gpu-short,gpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out
# GPU type is chosen at submit time (see slurm/submit_race.sh): --gres=gpu:h200:1 or gpu:a100:1
# Arguments are Hydra overrides for scripts/10_detect.py, e.g. max_frames=32
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/race.sh            # exits quietly if another GPU type already won

echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export UV_CACHE_DIR=/scratch/$USER/.cache/uv UV_PYTHON_INSTALL_DIR=/scratch/$USER/.local/uv-python
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python scripts/10_detect.py "$@"
