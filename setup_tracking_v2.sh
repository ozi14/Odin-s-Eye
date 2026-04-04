#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# setup_tracking_v2.sh — Install D4SM + SAM2.1 + DINOv2 dependencies
#
# Run once on Colab (or any CUDA machine) before 10_local_tracker_v2.py
# ──────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Odin's Eye v2 — Tracking Pipeline Setup"
echo "════════════════════════════════════════════════════════════"

# ── 1. Clone D4SM ────────────────────────────────────────────────
if [ ! -d "external/d4sm" ]; then
    echo "[1/5] Cloning D4SM repository..."
    mkdir -p external
    git clone --depth 1 https://github.com/alanlukezic/d4sm.git external/d4sm
else
    echo "[1/5] D4SM already cloned."
fi

# ── 2. Verify d4sm structure ─────────────────────────────────────
#   setup.py lives at external/d4sm/ (repo root), NOT external/d4sm/sam2/
#   The sam2/ subfolder is a Python package dir, not a standalone project.
if [ ! -f "external/d4sm/setup.py" ]; then
    echo "FATAL: external/d4sm/setup.py not found. Broken clone?"
    echo "       Try:  rm -rf external/d4sm && bash $0"
    exit 1
fi

# ── 3. Install D4SM's modified SAM2 package ──────────────────────
echo "[2/5] Installing SAM2 (D4SM fork with track_step)..."
pip install -e external/d4sm/ || {
    echo "FATAL: sam2 install failed. Check errors above."
    exit 1
}

# Verify import works
python -c "import sam2; print('  ✓ sam2 imported from', sam2.__file__)" || {
    echo "FATAL: sam2 installed but cannot import."
    exit 1
}

# ── 4. Download SAM2.1-Large checkpoint ──────────────────────────
mkdir -p checkpoints
if [ ! -f "checkpoints/sam2.1_hiera_large.pt" ]; then
    echo "[3/5] Downloading SAM2.1-Large checkpoint (~900MB)..."
    wget -q --show-progress -P checkpoints/ \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "[3/5] SAM2.1-Large checkpoint already exists."
fi

# ── 5. Build SAM2 custom CUDA ops (optional but faster) ─────────
echo "[4/5] Building SAM2 CUDA extensions..."
cd external/d4sm
python setup.py build_ext --inplace 2>/dev/null && \
    echo "  ✓ CUDA extensions built" || \
    echo "  ⚠ CUDA extensions skipped (tracking still works, some postprocessing disabled)"
cd "$SCRIPT_DIR"

# ── 6. Install remaining Python dependencies ─────────────────────
echo "[5/5] Installing Python dependencies..."
pip install -q hydra-core omegaconf ultralytics

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Setup complete. DINOv2 will auto-download on first run."
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Verify:  python -c \"import sam2; print(sam2.__file__)\""
echo ""