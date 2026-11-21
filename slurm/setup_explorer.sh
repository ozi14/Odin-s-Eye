#!/bin/bash
# One-time Explorer setup on the LOGIN node: uv, repo, environment, MOT20.  Safe to re-run.
set -euo pipefail
ROOT=/scratch/$USER/odins-eye
BRANCH=${BRANCH:-v4-phase1}
export UV_CACHE_DIR=/scratch/$USER/.cache/uv            # keep big caches out of the home quota
export UV_PYTHON_INSTALL_DIR=/scratch/$USER/.local/uv-python
export UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=8               # Explorer downloads from GitHub can stall

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [[ -d $ROOT/.git ]]; then git -C "$ROOT" pull --ff-only; else git clone -b "$BRANCH" https://github.com/ozi14/Odin-s-Eye.git "$ROOT"; fi
cd "$ROOT"
# Python 3.12: uv's own downloader stalls on Explorer, so fetch the same standalone build with curl
PYDIR=/scratch/$USER/.local/python3.12
PY_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16+20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
if [[ ! -x $PYDIR/bin/python3.12 ]]; then
  mkdir -p "$PYDIR" && curl -fL --retry 5 "$PY_URL" | tar -xz -C "$PYDIR" --strip-components=1
fi
uv sync --frozen --python "$PYDIR/bin/python3.12"     # exact versions from uv.lock (Linux, CUDA 13)
mkdir -p data/raw data/weights slurm/logs

if [[ ! -d data/raw/MOT20/train/MOT20-05 ]]; then
  curl -fL -o /scratch/$USER/MOT20.zip https://motchallenge.net/data/MOT20.zip     # ~5 GB
  unzip -q /scratch/$USER/MOT20.zip -d data/raw && rm /scratch/$USER/MOT20.zip
  chmod -R a-w data/raw/MOT20
fi
for s in 01 02 03 05; do printf "MOT20-%s: %s frames, gt md5 %s\n" $s \
  "$(ls data/raw/MOT20/train/MOT20-$s/img1 | wc -l)" "$(md5sum < data/raw/MOT20/train/MOT20-$s/gt/gt.txt | cut -c1-12)"; done
echo "setup done: $ROOT   (now upload data/weights/*.pt from the Mac)"
