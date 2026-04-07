# Odin's Eye v3 — Single-Camera Dense Pedestrian Tracking + VLM Narration

## Project Identity

- **Course:** CS 5330 Pattern Recognition and Computer Vision, Prof. Akram Bayat, Northeastern University
- **Team:** Taha Oğuzhan Uçar, Xinyi Jiang
- **Dataset:** MOT20 (4 train + 4 test sequences, dense crowds up to 150+ pedestrians/frame)
- **Hardware:** Apple M4 Max (dev), Colab A100/Blackwell (inference)

---

## Motivation & Pivot

The original Odin's Eye (v1/v2) targeted multi-camera tracking on WILDTRACK using SAM2.1 + D4SM + DINOv2 across 7 synchronized cameras. While the geometric calibration and D4SM memory management were technically sound, the pipeline suffered from:

1. **Detection recall bottleneck** — YOLO missed 30-40% of pedestrians in dense groups regardless of fine-tuning
2. **GPU memory pressure** — 7 cameras × SAM2.1-Large + DINOv2 ViT-L exceeded 95GB even with CPU offload
3. **Architecture mismatch** — D4SM was designed for VOT (few objects, clean scenes), not MOT (150+ objects, dense occlusion)

v3 pivots to **single-camera dense pedestrian tracking on MOT20** with a proven MOT tracker (ByteTrack) and adds **VLM-guided semantic narration** as the novel contribution.

---

## Core Idea

Track all pedestrians in dense crowd video using ByteTrack (detection + Kalman filter + byte-level association), enhance identity persistence with DINOv2 ReID features, then periodically invoke a Vision-Language Model (Qwen3-VL) to generate structured natural-language descriptions of tracked individuals and scene dynamics. The VLM narration layer transforms raw tracking telemetry into human-readable intelligence — clothing descriptions, behavioral analysis, interaction detection, anomaly flagging.

---

## System Architecture — 3-Phase Pipeline

### Phase 1: Dense Pedestrian Tracking (ByteTrack + DINOv2)

**Script:** `mot/scripts/01_track.py`

```
Frame → YOLO Detect → ByteTrack Associate → DINOv2 ReID → Tracked Identities
```

- **Detection:** YOLOv11x (or fine-tuned YOLO26m) at native resolution. Runs every frame (MOT20 requires it — people move fast in dense crowds). Confidence threshold 0.3 with aggressive NMS.
- **Tracking:** ByteTrack two-pass association:
  - First pass: match high-confidence detections (>0.6) to existing tracks using IoU + Kalman-predicted position
  - Second pass: match remaining low-confidence detections (0.3-0.6) to unmatched tracks — this is ByteTrack's key insight, recovering occluded people via weak detections
- **ReID Enhancement:** DINOv2 ViT-B (or ViT-L) extracts 768-D (or 1024-D) appearance features per detected person crop. Used as a secondary association signal when IoU-based matching is ambiguous (long occlusions, crossing paths). EMA-smoothed feature gallery per track.
- **Track Lifecycle:** New detections that don't match any track for 3 consecutive frames spawn a new identity. Tracks with no detection for 30 frames are terminated. Re-identification via DINOv2 cosine similarity can resurrect old tracks within a 100-frame window.
- **Output:** Per-frame JSON with track_id, bbox, confidence, ReID embedding. MOTChallenge-format txt for benchmark evaluation.

### Phase 2: VLM Scene Narration (Qwen3-VL)

**Script:** `mot/scripts/02_narrate.py`

```
Every N frames: Frame + Tracking Overlay → Qwen3-VL → Structured JSON Description
```

- **Model:** Qwen3-VL-4B-Instruct on Apple Silicon (MLX) or Colab GPU
- **Input:** Full frame with bounding boxes + track IDs overlaid, plus a text prompt describing what to narrate
- **Narration Modes:**
  - **Scene summary:** "Describe the overall scene — crowd density, movement patterns, notable events"
  - **Person description:** "Describe person T005 — clothing, accessories, posture, direction of movement"
  - **Interaction detection:** "Are any tracked people interacting? Describe the interaction"
  - **Anomaly flagging:** "Is anyone behaving unusually compared to the crowd flow?"
- **Output:** Structured JSON with per-person descriptions, scene summary, flagged events
- **Frequency:** Every 30 frames (1 second at 30fps) for scene summary, on-demand for person descriptions

### Phase 3: Benchmark & Evaluation

**Script:** `mot/scripts/03_benchmark.py`

- **MOT Metrics:** MOTA, MOTP, IDF1, HOTA using official `TrackEval` or `motmetrics` library
- **VLM Evaluation:** Manual qualitative assessment of narration accuracy (clothing match, interaction correctness)
- **Comparison:** ByteTrack baseline vs ByteTrack + DINOv2 ReID vs full pipeline with VLM

---

## Dataset: MOT20

| Sequence | Frames | Resolution | Avg Pedestrians/Frame | Scene |
|----------|--------|------------|----------------------|-------|
| MOT20-01 | 429 | 1920×1080 | ~75 | Night, indoor |
| MOT20-02 | 2782 | 1920×1080 | ~150 | Day, outdoor square |
| MOT20-03 | 2405 | 1920×1080 | ~125 | Night, outdoor |
| MOT20-05 | 3315 | 1920×1080 | ~150 | Day, outdoor square |

Train sequences have ground-truth annotations. Test sequences are for MOTChallenge submission.

Source: https://www.kaggle.com/datasets/ismailelbouknify/mot-20

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **Detection** | YOLOv11x or YOLO26m | Best speed/accuracy for dense pedestrian detection |
| **Tracking** | ByteTrack | Proven MOT20 SOTA, handles 150+ objects, byte-level low-conf recovery |
| **ReID** | DINOv2 ViT-B/L | Self-supervised, cross-domain robust, strong appearance features |
| **Mask (selective)** | SAM2.1 | Only for VLM-narrated targets, not bulk tracking |
| **VLM** | Qwen3-VL-4B-Instruct | Runs on Apple Silicon via MLX, structured output capable |
| **Evaluation** | TrackEval / motmetrics | Standard MOTChallenge evaluation |

---

## Folder Structure

```
mot/
├── docs/
│   └── PROJECT_CONTEXT.md      ← this file
├── scripts/
│   ├── 00_download_mot20.py    ← dataset download + setup
│   ├── 01_track.py             ← Phase 1: ByteTrack + DINOv2 tracking
│   ├── 02_narrate.py           ← Phase 2: VLM narration
│   └── 03_benchmark.py         ← Phase 3: MOT metrics evaluation
├── odin_eye_mot/
│   ├── __init__.py
│   ├── tracker/
│   │   ├── __init__.py
│   │   ├── bytetrack.py        ← ByteTrack core algorithm
│   │   └── kalman_filter.py    ← Kalman filter for motion prediction
│   ├── reid/
│   │   ├── __init__.py
│   │   └── dinov2_extractor.py ← DINOv2 ReID (reused from v2)
│   └── vlm/
│       ├── __init__.py
│       └── narrator.py         ← Qwen3-VL narration engine
├── output/                     ← tracking results, narration JSONs
├── datasets/                   ← MOT20 data (not committed)
└── requirements.txt
```

---

## What We Keep From v1/v2

- **DINOv2 ReID extractor** (`dinov2_extractor.py`) — reused directly, proven feature quality
- **YOLO detection pipeline** — detection code, confidence filtering, NMS
- **Benchmark methodology** — MOTA/MOTP computation approach
- **VLM narration concept** — the novel layer, now actually implemented

## What We Drop

- Multi-camera calibration (Phase 0) — not needed for single camera
- SAM2.1 + D4SM for bulk tracking — replaced by ByteTrack for MOT
- Ground-plane homography/projection — not applicable to MOT20
- Global cross-camera association (Phase 2 old) — single camera, not needed
- CPU offload complexity — ByteTrack uses negligible memory

## What's New

- **ByteTrack** — purpose-built MOT tracker for dense scenes
- **VLM narration pipeline** — the main novel contribution
- **MOT20 dataset** — standard benchmark, meaningful comparison possible
- **Selective SAM2 masking** — masks only for VLM-described targets