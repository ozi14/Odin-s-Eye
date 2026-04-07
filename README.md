# Odin's Eye v3 — Dense Pedestrian Tracking + VLM Narration

> Single-camera MOT on MOT20 with ByteTrack, DINOv2 ReID, and Qwen3-VL scene narration.

**Course:** CS 5330 Pattern Recognition & Computer Vision, Prof. Akram Bayat, Northeastern University  
**Team:** Taha Oğuzhan Uçar, Xinyi Jiang

---

## Overview

Odin's Eye v3 tracks all pedestrians in dense crowd video using **ByteTrack** (detection + Kalman filter + byte-level association), enhances identity persistence with **DINOv2** ReID features, then invokes **Qwen3-VL** to generate structured natural-language descriptions of the scene.

### Pipeline

```
Phase 1                        Phase 2                    Phase 3
YOLO → ByteTrack + DINOv2  →  Qwen3-VL Narration  →  MOT Metrics Eval
       (Tracking)                (Scene Description)     (MOTA/IDF1)
```

---

## Quick Start

```bash
# 1. Activate environment
source dump/bin/activate

# 2. Install dependencies
pip install -r mot/requirements.txt

# 3. Download MOT20 dataset
python mot/scripts/00_download_mot20.py

# 4. Run tracking (Phase 1)
cd mot
python scripts/01_track.py --seq MOT20-01 --visualize

# 5. Evaluate (Phase 3)
python scripts/03_benchmark.py --seq MOT20-01 --save_json

# 6. Run VLM narration (Phase 2)
python scripts/02_narrate.py --seq MOT20-01 --backend mlx --narrate_every 60
```

---

## Results (MOT20-01)

| Metric | Value |
|--------|-------|
| **MOTA** | 70.92% |
| **IDF1** | 69.12% |
| **ID Switches** | 63 |
| **Precision** | 90.28% |
| **Recall** | 79.83% |
| **Unique IDs** | 95 (GT: 74) |

---

## Tech Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Detection | YOLO26m (CrowdHuman fine-tuned) | Dense pedestrian detection |
| Tracking | ByteTrack (custom, 608 LOC) | Two-pass association + Kalman filter |
| ReID | DINOv2 ViT-B/14 (768-D) | Appearance features, EMA gallery |
| VLM | Qwen3-VL-4B-Instruct-4bit (MLX) | Scene narration, anomaly detection |
| Evaluation | motmetrics | MOTA, MOTP, IDF1 |

---

## Project Structure

```
CV_term_project/
├── mot/                           ← active project
│   ├── scripts/
│   │   ├── 00_download_mot20.py   # dataset download
│   │   ├── 01_track.py            # Phase 1: tracking
│   │   ├── 02_narrate.py          # Phase 2: VLM narration
│   │   └── 03_benchmark.py        # Phase 3: evaluation
│   ├── odin_eye_mot/              # library package
│   │   ├── tracker/
│   │   │   ├── bytetrack.py       # ByteTrack + ReID resurrection
│   │   │   └── kalman_filter.py   # 8-D constant-velocity KF
│   │   ├── reid/
│   │   │   └── dinov2_extractor.py
│   │   └── vlm/
│   │       └── narrator.py        # Qwen3-VL narration engine
│   ├── datasets/MOT20/            # MOT20 data (not committed)
│   ├── output/                    # tracking & narration results
│   └── requirements.txt
├── models/                        # YOLO weights
├── dump/                          # Python virtual environment
└── old/                           # legacy v1/v2 code
```

---

## Key Features

- **ByteTrack Two-Pass Association** — recovers occluded pedestrians via low-confidence detections
- **DINOv2 ReID Resurrection** — re-identifies removed tracks via appearance matching within a 100-frame window
- **4 VLM Narration Modes** — scene summary, person description, interaction detection, anomaly flagging
- **Dual VLM Backends** — MLX (Apple Silicon) and HuggingFace Transformers (CUDA)

---

## Configuration

Key CLI parameters for `01_track.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf_thresh` | 0.3 | YOLO confidence threshold |
| `--thresh_high` | 0.6 | ByteTrack high-conf cutoff |
| `--thresh_low` | 0.1 | ByteTrack low-conf cutoff |
| `--max_lost_frames` | 30 | Frames before track removal |
| `--reid_weight` | 0.3 | IoU vs ReID blend (0–1) |
| `--no_reid` | — | Disable DINOv2 ReID |
| `--visualize` | — | Save annotated frames |

---

## Hardware

- **Development:** Apple M4 Max (MPS + MLX)
- **Inference:** Google Colab A100 / Blackwell

---

## Dataset

[MOT20](https://www.kaggle.com/datasets/ismailelbouknify/mot-20) — 4 train + 4 test sequences, dense crowds up to 150+ pedestrians/frame, 1920×1080 resolution.

---

## License

MIT
