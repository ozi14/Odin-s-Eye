# Odin's Eye: Dense Pedestrian Tracking with Vision-Language Scene Narration

> A single-camera multi-object tracking system on MOT20 with ByteTrack, DINOv2 ReID, and Qwen3-VL scene narration.

**Course:** CS 5330 Pattern Recognition & Computer Vision, Prof. Akram Bayat, Northeastern University  
**Team:** Taha Oğuzhan Uçar, Xinyi Jiang

---

## Purpose

Dense crowd monitoring is a critical challenge in computer vision, where frequent occlusions, visually similar appearances, and high pedestrian counts make reliable tracking extremely difficult. While state-of-the-art trackers produce numerical bounding box trajectories, these raw outputs lack the semantic interpretation needed by human operators in surveillance, urban planning, and public safety applications.

**Odin's Eye** bridges this gap by combining robust dense pedestrian tracking with automated vision-language scene narration. The system:

1. **Tracks** all pedestrians in dense crowd video using ByteTrack with DINOv2 appearance features
2. **Narrates** the scene by periodically invoking a Vision-Language Model (Qwen3-VL) to produce structured descriptions of crowd dynamics, individual appearances, interactions, and anomalies
3. **Evaluates** tracking quality against MOT20 ground truth using standard MOTChallenge metrics

The result is a pipeline that transforms raw video into both quantitative tracking data and human-readable scene intelligence.

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
| **Fragmentations** | 196 |

---

## Project Structure

```
CV_term_project/
├── README.md                          ← you are here
├── .gitignore
├── LICENSE
├── models/                            ← YOLO weights (not committed)
│   └── yolo26_weights_v1/
│       └── best.pt                    # CrowdHuman-finetuned YOLO26m
├── dump/                              ← Python virtual environment
├── mot/                               ← active project
│   ├── CONTEXT.md                     # design document & project context
│   ├── requirements.txt               # Python dependencies
│   ├── scripts/                       # executable pipeline scripts
│   │   ├── 00_download_mot20.py
│   │   ├── 01_track.py
│   │   ├── 02_narrate.py
│   │   └── 03_benchmark.py
│   ├── odin_eye_mot/                  # library package
│   │   ├── __init__.py
│   │   ├── tracker/
│   │   │   ├── __init__.py
│   │   │   ├── bytetrack.py
│   │   │   └── kalman_filter.py
│   │   ├── reid/
│   │   │   ├── __init__.py
│   │   │   └── dinov2_extractor.py
│   │   └── vlm/
│   │       ├── __init__.py
│   │       └── narrator.py
│   ├── datasets/                      ← MOT20 data (not committed)
│   │   └── MOT20/
│   │       ├── train/ (MOT20-01..05)
│   │       └── test/  (MOT20-04..08)
│   └── output/                        ← tracking & narration results
│       └── MOT20-01/
│           ├── mot_results.txt        # MOTChallenge format output
│           ├── metrics.json           # benchmark results
│           ├── frames/                # per-frame JSON track data
│           ├── vis/                   # annotated frame images
│           └── narration/             # VLM narration JSONs
└── old/                               ← legacy v1/v2 code (separate branch)
```

---

## Environment Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Apple Silicon Mac** (M-series) for local VLM inference via MLX, or **CUDA GPU** for Transformers backend
- **Git** for version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/ozi14/Odin-s-Eye.git
cd Odin-s-Eye
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv dump
source dump/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r mot/requirements.txt
```

This installs the following dependency groups:

| Category | Packages | Purpose |
|----------|----------|---------|
| **Core** | `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, `scipy`, `Pillow`, `tqdm`, `matplotlib` | Detection, tensor ops, image processing |
| **Tracking Evaluation** | `motmetrics` | MOTChallenge metric computation (MOTA, IDF1, etc.) |
| **VLM — Apple Silicon** | `mlx`, `mlx-vlm` | Local Qwen3-VL inference via MLX framework |
| **VLM — CUDA GPU** | `transformers`, `accelerate`, `qwen-vl-utils` | HuggingFace-based VLM inference |
| **Dataset** | `kagglehub`, `huggingface_hub` | Automated dataset downloading |

### Step 4: Download MOT20 Dataset

```bash
cd mot
python scripts/00_download_mot20.py
```

This downloads the MOT20 dataset from Kaggle and structures it into `mot/datasets/MOT20/` with the standard MOTChallenge layout.

### Step 5: Obtain YOLO Weights

Place your YOLO model weights at `models/yolo26_weights_v1/best.pt`. If not available, the tracking script will automatically fall back to pretrained `yolo11x.pt` (downloaded by Ultralytics on first run).

---

## Running the Pipeline

All scripts are run from the `mot/` directory:

```bash
cd mot
```

### Phase 1: Tracking

```bash
# Basic run on one sequence
python scripts/01_track.py --seq MOT20-01

# With visualization output
python scripts/01_track.py --seq MOT20-01 --visualize

# Multiple sequences
python scripts/01_track.py --seq MOT20-01 MOT20-02 MOT20-03 MOT20-05

# Quick test (first 50 frames only)
python scripts/01_track.py --seq MOT20-01 --max_frames 50

# Disable ReID (IoU-only baseline for ablation)
python scripts/01_track.py --seq MOT20-01 --no_reid

# Tune ReID strength
python scripts/01_track.py --seq MOT20-01 --reid_weight 0.5 --max_lost_frames 60
```

**Output:** `output/<seq>/mot_results.txt`, `output/<seq>/frames/`, optionally `output/<seq>/vis/`

### Phase 2: VLM Narration

```bash
# Scene summary narration (Apple Silicon)
python scripts/02_narrate.py --seq MOT20-01 --backend mlx --narrate_every 60

# Person-specific description
python scripts/02_narrate.py --seq MOT20-01 --backend mlx --mode person_describe --highlight_id 5

# Anomaly detection mode
python scripts/02_narrate.py --seq MOT20-01 --backend mlx --mode anomaly

# CUDA GPU backend (Colab)
python scripts/02_narrate.py --seq MOT20-01 --backend transformers --device cuda
```

**Output:** `output/<seq>/narration/narration_XXXXXX.json`, `output/<seq>/narration/narration_summary.json`

### Phase 3: Benchmark Evaluation

```bash
# Evaluate one sequence
python scripts/03_benchmark.py --seq MOT20-01 --save_json

# Evaluate all train sequences
python scripts/03_benchmark.py --seq MOT20-01 MOT20-02 MOT20-03 MOT20-05 --save_json

# Compare ablation (e.g., no-ReID variant)
python scripts/03_benchmark.py --seq MOT20-01 --pred_suffix _no_reid
```

**Output:** Terminal metrics report + `output/<seq>/metrics.json`

---

## Scripts

### `00_download_mot20.py` — Dataset Download
Downloads the MOT20 dataset from Kaggle via `kagglehub`, auto-detects the nested directory structure, and copies it into the standard MOTChallenge layout at `mot/datasets/MOT20/`. Supports `--skip_download` to verify an existing installation.

### `01_track.py` — Phase 1: Dense Pedestrian Tracking
The main tracking script. Reads MOT20 sequences frame-by-frame, runs YOLO detection (person class only, confidence threshold 0.3), and feeds detections into a ByteTracker instance for two-pass association. Optionally extracts DINOv2 ReID features for appearance-based matching. Outputs MOTChallenge-format text files, per-frame JSON track data, and optional annotated visualization frames.

### `02_narrate.py` — Phase 2: VLM Scene Narration
Reads Phase 1 tracking output (frame JSONs) and original images, draws bounding boxes with track IDs onto frames, and sends annotated images to Qwen3-VL for structured narration. Supports four modes: scene summary (crowd density and movement), person description (clothing and posture for a specific track), interaction detection (person-to-person), and anomaly flagging (unusual behavior). Outputs structured JSON narrations at configurable frame intervals.

### `03_benchmark.py` — Phase 3: MOT Metrics Evaluation
Loads ground-truth annotations and predicted tracks in MOTChallenge format, then computes standard metrics using the `motmetrics` library: MOTA, MOTP, IDF1, identity switches, fragmentations, precision, recall, mostly tracked (MT), and mostly lost (ML). Supports multi-sequence evaluation with summary tables and JSON export for downstream analysis.

---

## Modules

### `odin_eye_mot/tracker/bytetrack.py` — ByteTrack Core Algorithm
Implements the ByteTrack two-pass association algorithm (Zhang et al., ECCV 2022) with integrated DINOv2 ReID. Manages track lifecycle (New → Tracked → Lost → Removed) with Kalman filter motion prediction. Key features include: two-pass detection association (high-confidence then low-confidence), blended IoU + ReID cosine similarity cost matrices, EMA-smoothed appearance gallery per track, and a ReID resurrection mechanism that recovers terminated tracks via appearance matching within a configurable window.

### `odin_eye_mot/tracker/kalman_filter.py` — Kalman Filter
An 8-dimensional constant-velocity Kalman filter for bounding box motion prediction. State vector: `[cx, cy, w, h, v_cx, v_cy, v_w, v_h]`. Process noise is scaled by bounding box dimensions following the ByteTrack reference implementation. Provides `initiate()`, `predict()`, and `update()` operations called once per track per frame.

### `odin_eye_mot/reid/dinov2_extractor.py` — DINOv2 ReID Feature Extractor
Extracts L2-normalized appearance feature vectors from person crops using DINOv2 ViT-B/14 (768-D features) loaded via `torch.hub`. Supports single and batch extraction with optional foreground masking. Input images are resized to 224×224 and ImageNet-normalized. Auto-detects compute device (CUDA → MPS → CPU). Reused from the v2 pipeline with proven feature quality.

### `odin_eye_mot/vlm/narrator.py` — Qwen3-VL Narration Engine
Orchestrates VLM-based scene narration with two inference backends: MLX (Apple Silicon, Qwen3-VL-4B-Instruct-4bit) and HuggingFace Transformers (CUDA, Qwen2-VL-7B-Instruct). Builds annotated frames with bounding boxes and track IDs, constructs mode-specific prompts, invokes the VLM, and parses structured JSON from model output with regex fallback for robust extraction. Narration frequency is configurable (default: every 30 frames).

---

## Configuration Reference

### Tracking Parameters (`01_track.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq` | *(required)* | Sequence name(s), e.g., `MOT20-01 MOT20-02` |
| `--conf_thresh` | `0.3` | YOLO confidence threshold |
| `--imgsz` | `1280` | YOLO inference image size |
| `--thresh_high` | `0.6` | ByteTrack high-confidence cutoff (Pass 1) |
| `--thresh_low` | `0.1` | ByteTrack low-confidence cutoff (Pass 2) |
| `--max_lost_frames` | `30` | Frames before a lost track is removed |
| `--reid_weight` | `0.3` | IoU vs ReID blend weight (0 = IoU only, 1 = ReID only) |
| `--reid_thresh` | `0.3` | Minimum cosine similarity for ReID match |
| `--reid_model` | `dinov2_vitb14_reg` | DINOv2 model variant (vitb, vitl, vits) |
| `--no_reid` | `false` | Disable DINOv2 ReID entirely |
| `--visualize` | `false` | Save annotated frame images |
| `--max_frames` | `None` | Process only first N frames (for testing) |
| `--device` | auto | Compute device: `cuda`, `mps`, or `cpu` |

### Narration Parameters (`02_narrate.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seq` | *(required)* | Sequence name |
| `--backend` | `mlx` | VLM backend: `mlx` (Apple Silicon) or `transformers` (CUDA) |
| `--mode` | `scene_summary` | Narration mode: `scene_summary`, `person_describe`, `interaction`, `anomaly` |
| `--narrate_every` | `30` | Generate narration every N frames |
| `--highlight_id` | `None` | Track ID to highlight (for `person_describe` mode) |
| `--model_id` | `None` | Override default VLM model ID |

---

## Tech Stack

| Component | Model / Library | Purpose |
|-----------|----------------|---------|
| **Detection** | YOLO26m (CrowdHuman fine-tuned) | Dense pedestrian detection |
| **Tracking** | ByteTrack (custom, 608 LOC) | Two-pass association + Kalman filter |
| **ReID** | DINOv2 ViT-B/14 (768-D) | Appearance features with EMA gallery |
| **VLM** | Qwen3-VL-4B-Instruct-4bit (MLX) | Scene narration and anomaly detection |
| **Evaluation** | motmetrics | MOTA, MOTP, IDF1, ID switches |
| **Dataset** | MOT20 (Kaggle) | 4 train + 4 test sequences, up to 150 ped/frame |

---

## Hardware

| Environment | Specification | Use |
|-------------|---------------|-----|
| **Development** | Apple M4 Max, MPS + MLX | Tracking, local VLM inference |
| **Training / Inference** | Google Colab A100 / Blackwell | Large-scale tracking, Transformers VLM backend |

---

## Dataset

### Tracking Benchmark: MOT20
[MOT20](https://www.kaggle.com/datasets/ismailelbouknify/mot-20) — A benchmark for multi-object tracking in dense crowd scenarios. Contains 4 training sequences with ground-truth annotations and 4 test sequences for MOTChallenge server evaluation. All sequences are 1920×1080 resolution at 25fps, featuring indoor and outdoor scenes with up to 150+ pedestrians per frame.

| Sequence | Frames | Avg Pedestrians/Frame | Scene |
|----------|--------|-----------------------|-------|
| MOT20-01 | 429 | ~75 | Indoor, night |
| MOT20-02 | 2,782 | ~150 | Outdoor square, day |
| MOT20-03 | 2,405 | ~125 | Outdoor, night |
| MOT20-05 | 3,315 | ~150 | Outdoor square, day |

### YOLO Fine-Tuning: CrowdHuman + CCTV
To detect pedestrians robustly across various views and occlusions, the `YOLO26m` detector was fine-tuned on a merged dataset containing crowd scenarios and CCTV surveillance camera footage.
- **Dataset Link:** [CrowdHuman+CCTV Merged Dataset (Google Drive)](https://drive.google.com/drive/folders/1oPijfK5Jo5GJiMRCebIm_CFVfWLz_fJG?usp=sharing)

---

## License

MIT
