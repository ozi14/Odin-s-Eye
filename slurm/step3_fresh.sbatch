#!/bin/bash
# Fresh Explorer run of Step 3 on ONE GPU node: setup -> verify -> smoke test -> full detection.
# Submit from /scratch/$USER/staging (holds this file + the weights), e.g.:
#   rm -rf race_winner; for g in h200 a100; do sbatch --gres=gpu:$g:1 step3_fresh.sbatch; done
# (first to start wins). Re-runs reuse the clone, the uv cache and MOT20 on /scratch.
#SBATCH --job-name=oe-step3
#SBATCH --partition=gpu-short,gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
set -euo pipefail
STAGE=/scratch/$USER/staging
ROOT=/scratch/$USER/odins-eye
WEIGHTS=yolo26m_ft1_v3_1280.pt
WEIGHTS_SHA=379b9bbdbf0f91fb57fa87306c42401f2e7591c152066ee76c003eb3e7927e87
EXPECTED="MOT20-01:429:b392a84b4eec8c07947dcb1fb089a48e MOT20-02:2782:b27e6cf03f80a1fb6e477899bd16efe2 MOT20-03:2405:1849781121d01039245b849609165389 MOT20-05:3315:3017ae96ab21cd34506c297be8f93ef3"

# --- race: the first job (any GPU type) to start wins and cancels the others -----------------
if mkdir "$STAGE/race_winner" 2>/dev/null; then
  echo "$SLURM_JOB_ID" > "$STAGE/race_winner/job"
  for j in $(squeue -h -u "$USER" -n oe-step3 -t PD -o %i); do scancel "$j" && echo "cancelled sibling $j"; done
else
  echo "sibling $(cat "$STAGE/race_winner/job") already running; exiting"; exit 0
fi

T0=$SECONDS
stage() { echo; echo "=== [$(date +%T)] +$(( (SECONDS - T0) / 60 )) min  $*"; }
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/scratch/$USER/.cache/uv UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8

stage "0 node"
echo "job $SLURM_JOB_ID on $(hostname), partition $SLURM_JOB_PARTITION, $SLURM_CPUS_PER_TASK CPUs"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
curl -sfI https://pypi.org > /dev/null && echo "internet: ok"

stage "1 clone + environment + MOT20 (slurm/setup_explorer.sh)"
if [[ -d $ROOT/.git ]]; then git -C "$ROOT" pull -q --ff-only; else git clone -q -b v4-phase1 https://github.com/ozi14/Odin-s-Eye.git "$ROOT"; fi
git -C "$ROOT" log --oneline -1
bash "$ROOT/slurm/setup_explorer.sh"
cd "$ROOT"

stage "2 weights"
cp "$STAGE/$WEIGHTS" data/weights/
echo "$WEIGHTS_SHA  data/weights/$WEIGHTS" | sha256sum -c -

stage "3 verify MOT20 is byte-identical to the Mac copy"
for e in $EXPECTED; do
  IFS=: read -r seq n md5 <<< "$e"
  got_n=$(ls "data/raw/MOT20/train/$seq/img1" | wc -l); got_md5=$(md5sum < "data/raw/MOT20/train/$seq/gt/gt.txt" | cut -d' ' -f1)
  [[ $got_n == "$n" && $got_md5 == "$md5" ]] && echo "$seq ok ($n frames)" || { echo "$seq MISMATCH: $got_n $got_md5"; exit 1; }
done

stage "4 verify environment: CUDA + test suite"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
python scripts/00_prepare_data.py
pytest -q

stage "5 smoke test: 32 frames of MOT20-01"
python scripts/10_detect.py max_frames=32 'data.sequences=[MOT20-01]'

stage "6 STEP 3: detection on all four MOT20 train sequences"
python scripts/10_detect.py

stage "done"
du -sh data/interim/detections/*
