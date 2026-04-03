#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# setup_tracking_v2.sh — Install D4SM + SAM2.1 + DINOv2 dependencies
#
# Run once on Colab (or any CUDA machine) before using scripts/pipeline/10_local_tracker_v2.py
# ──────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Odin's Eye v2 — Tracking Pipeline Setup"
echo "════════════════════════════════════════════════════════════"

# 1. Clone D4SM (contains modified SAM2 with track_step support)
if [ ! -d "external/d4sm" ]; then
    echo "[1/4] Cloning D4SM repository..."
    mkdir -p external
    git clone --depth 1 https://github.com/alanlukezic/d4sm.git external/d4sm
else
    echo "[1/4] D4SM already cloned."
fi

# 2. Install D4SM's modified SAM2 package
echo "[2/4] Installing SAM2 (D4SM fork with track_step)..."
pip install -e external/d4sm/sam2/ 2>&1 | tail -1

# 3. Download SAM2.1-Large checkpoint
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "[3/4] Downloading SAM2.1-Large checkpoint (~900MB)..."
    wget -q --show-progress -P checkpoints/ \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "[3/4] SAM2.1-Large checkpoint already exists."
fi

# 4. Install Python dependencies
echo "[4/4] Installing Python dependencies..."
pip install -q hydra-core omegaconf ultralytics

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete. DINOv2 will auto-download on first run."
echo "════════════════════════════════════════════════════════════"
