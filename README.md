# Odin's Eye

Multi-camera person tracking on WILDTRACK using SAM2.1 + D4SM memory management, DINOv2 ReID, and VLM scene narration.

## Repository layout

| Path | Purpose |
|------|---------|
| `odin_eye/` | Core Python package: `reid/` (DINOv2 ViT-L), `tracking/` (D4SM / SAM2.1 engine) |
| `scripts/data_prep/` | Dataset prep and YOLO training helpers (`01_*` … `07_*`) |
| `scripts/pipeline/` | Pipeline scripts: calibration (`09`), local tracking (`10`), global ID (`11`), benchmark (`12`) |
| `scripts/tools/` | Utilities (e.g. `make_video.py`) |
| `docs/` | Design notes, calibration doc, and project context |
| `old/` | Archived v1 code: legacy tracker, OSNet-AIN, BoT-SORT config, early VLM experiments |
| `datasets/` | Data root *(not committed)* |
| `external/` | Cloned deps e.g. D4SM *(created by `setup_tracking_v2.sh`; not committed)* |
| `checkpoints/` | SAM2.1 weights *(not committed)* |
| `models/` | YOLO weights *(not committed)* |

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
```

### Phase 0 — Calibration (runs on any machine)

```bash
python scripts/pipeline/09_calibration.py
```

### Phase 1 — Local tracking (requires CUDA for SAM2.1)

```bash
bash setup_tracking_v2.sh                    # clone D4SM, install SAM2, download checkpoint
python scripts/pipeline/10_local_tracker_v2.py --device cuda --split val
```

### Phase 2 — Global association

```bash
python scripts/pipeline/11_global_association.py
```

## Colab

Clone the repo, install deps, run `setup_tracking_v2.sh`, then stage datasets under `datasets/` on the VM disk (avoid reading from Google Drive for every frame).
