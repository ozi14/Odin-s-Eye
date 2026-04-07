# Odin's Eye — Final Project Context
## Cooperative Multi-Camera Tracking & VLM-Guided Semantic Narration

---

## Project Identity

- **Course:** CS 5330 Pattern Recognition and Computer Vision, Prof. Akram Bayat, Northeastern University
- **Team:** Taha Oğuzhan Uçar (ucar.t@northeastern.edu), Xinyi Jiang (jiang.x2@northeastern.edu)
- **Hardware:** Apple M4 Max, 48GB unified memory, MPS/MLX backend (local training/inference)
- **Dataset:** WILDTRACK (7 synchronized static HD cameras, 36m × 12m outdoor courtyard)

---

## Core Idea

"Odin's Eye" is a multimodal multi-camera surveillance pipeline combining precise geometric calibration, mask-level object tracking, domain-generalized ReID appearance matching, and Large Vision-Language Models (VLMs) to track individuals across camera views and generate natural-language intelligence from video streams. The v2 architecture replaces bbox-level BoT-SORT tracking with SAM2.1-Large + D4SM distractor-aware memory management, upgrades ReID from OSNet-AIN (512-D) to DINOv2 ViT-L (1024-D), and runs heavy inference on Colab A100 while keeping VLM narration local on Apple Silicon.

---

## Final System Architecture — 4-Phase Pipeline

### Phase 0: Offline Calibration & Setup (One-Time)
Transforms raw camera parameter files into workable spatial geometry mappings.
- **Matrix Computation:** Extracts intrinsics ($K$), rotation ($R$), and translation ($t$) from WILDTRACK XMLs to compute Projection matrices ($P = K[R|t]$).
- **Homography Generation:** Calculates inverse homographies ($H^{-1}$) to project 2D image coordinates (foot-points) onto the 3D ground plane ($Z=0$).
- **FOV Geometry & Overlaps:** Uses ground-plane forward projection to compute the exact Field of View polygons for all cameras, circumventing horizon-line issues in steep-angle cameras.
- **Outputs:** Overlap matrix, BEV (Birds-Eye-View) layouts, and robust mapping parameters cached in `calibration_cache.json`.

### Phase 1 v2: SAM2.1 + D4SM Mask Tracking (`scripts/pipeline/10_local_tracker_v2.py`)
Replaces YOLO+BoT-SORT+OSNet with a detection-guided SAM2.1 mask tracking pipeline.
- **SAM2.1-Large Backbone:** Hiera ViT-L image encoder (224M params) with memory attention. One model instance shared across all 7 cameras via separate `PerCameraState` objects.
- **D4SM Memory Management:** Distractor-Aware Memory split into RAM (Recent Appearance Memory, 3 slots) and DRM (Distractor Resolving Memory, 3 slots). Introspection via SAM2's multi-mask decoder detects distractors by comparing the predicted mask against alternative candidates.
- **Detection-Guided Tracking Loop:** YOLO runs every K frames as a person proposal generator. Unmatched detections initialise new D4SM tracks via SAM2's prompt encoder. Between detection frames, D4SM tracks all objects using the memory bank. Objects invisible for >T frames are removed.
- **Mask Foot-Point Projection:** Ground-plane positions computed from the bottom-centre of the segmentation mask contour (more accurate than bbox bottom under occlusion).
- **DINOv2 ViT-L ReID:** 1024-D L2-normalised features extracted from mask-cropped person images (background zeroed). Batched in groups of 64.
- **Outputs:** Same JSON+NPZ contract as v1 (`tracking_results_v2/`), with 1024-D embeddings.

### Phase 1 v1 (legacy): YOLO + BoT-SORT + OSNet (`legacy/10_local_tracker.py`)
Original bbox-level pipeline retained for comparison. See `tracking_results/`.

### Phase 2: Global Association (Cross-Camera Matching)
Fuses the multi-camera local tracks into persistent global identities.
- **Multimodal Fusion Engine:** Links tracklets using the Hungarian Algorithm based on a combined cost matrix.
- **Cost Formulation (v2 tuning):** 50/50 spatial + appearance weight (was 70/30). DINOv2 features are reliable enough to carry equal weight. Spatial gate tightened to 300cm (was 400cm) thanks to mask foot-point accuracy. Appearance gate tightened to 0.45 cosine distance (was 0.60).

### Phase 3: Semantic VLM Narration & Intelligence
Translates numeric scene state into contextual natural language.
- **Model:** **Qwen3-VL-4B-Instruct** running aggressively optimized on Apple Silicon via MLX (`mlx_vlm`).
- **Full-Frame Approach:** Instead of narrating per-crop (which scales poorly with crowd density), the VLM processes the entire frame with an overlay highlighting overlap regions or specific tracked targets.
- **Asynchronous Execution:** Because VLM inference is the pipeline bottleneck, narration runs asynchronously or cycles between cameras, periodically enriching the fast tracker telemetry with deep semantic descriptions (clothing, interactions, anomalies) formatted as structured JSON.

---

## Tech Stack & Dependencies

| Component | v1 | v2 |
|---|---|---|
| **Detection** | YOLOv26m (every frame) | YOLOv26m (every K frames, proposal generator) |
| **Tracking** | BoT-SORT (bbox) | SAM2.1-Large + D4SM (mask, distractor-aware) |
| **ReID** | OSNet-AIN 512-D (MPS) | DINOv2 ViT-L 1024-D (CUDA) |
| **Geometry** | OpenCV + Shapely | Same |
| **VLM Narrator** | Qwen3-VL-4B (MLX) | Pending Phase 3 integration |
| **Compute** | M4 Max only | M4 Max (dev) + Colab A100 (inference) |

---

## Evolution & Design Pivots

1. **Sequential vs. Parallel Local Tracking:** Parallel loading of 7 YOLO instances bloated GPU memory. Sequential processing cut memory pressure and frame time from ~3.5s to ~1.5s.
2. **Early Ground-Plane Discards:** Projecting foot-points before ReID eliminated ~20% of anomalous detections.
3. **Full-Frame vs Per-Crop VLM:** VLM encoding scales by image pass. Full-frame description replaced per-crop narration.
4. **v1 → v2 Architecture Pivot:** The v1 pipeline (YOLO+BoT-SORT+OSNet) achieved only 8.3% identity consistency (MODA=-1.807) due to weak OSNet cross-view features (cosine sim ~0.41 for same person front/back) and bbox foot-point projection errors (up to 3.7m). Inspired by DAM4SAM (VOTS2025 winner), v2 replaces the entire tracking backbone with SAM2.1-Large + D4SM distractor-aware memory + DINOv2 ViT-L ReID, targeting 40-60% identity consistency.

---

## Current Status

- **Phase 0:** Complete. Calibration cached.
- **Phase 1 v1:** Complete. Baseline results available.
- **Phase 1 v2:** Implemented (`scripts/pipeline/10_local_tracker_v2.py`). Requires Colab A100 for inference.
- **Phase 2:** Updated for DINOv2 features. Re-tuned gates for v2.
- **Phase 3:** Pending integration.
