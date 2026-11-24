# ADR-003: Detector for Phase 1 = yolo26m_ft1_v3_1280, frozen settings, detections-1.0

- Status: accepted · 2026-11-20 · updated 2026-11-23 (official run on Mac, NMS/clipping, torch pin) · owner: Taha
- Code: `src/odins_eye/perception/detector.py`, `scripts/10_detect.py`,
  config `configs/perception/detector/yolo26m_ft1_v3_1280.yaml`
- Results: [`docs/results/step3/`](../results/step3/README.md)

## Decision
- Weights `yolo26m_ft1_v3_1280.pt` (sha256 379b9bbd…): YOLO26m fine-tuned on CrowdHuman + CCTV
  Persons (no MOT17/MOT20 frames), long Explorer run, epoch 160, imgsz 1280. Validation
  (CrowdHuman+CCTV val): P .882, R .850, mAP50 .922, mAP50-95 .667; best of all saved
  checkpoints (ft1_v1 .649, ft1_v2 .662).
- Settings: imgsz 1280 (= training), conf 0.1, NMS IoU 0.7 (the head is not end-to-end, so NMS
  runs), max_det 1000 (default 300 could truncate dense frames), fp16 on CUDA / fp32 on MPS,
  class person only.
- Detection runs once per sequence; every tracker reads the same `detections-1.0` files.

## Official run (2026-11-23)
- **Official Step 3 detections = Mac run** at commit `41156db` (clean): MPS, fp32, torch 2.14.1,
  ultralytics 8.4.216 → `data/interim/detections/yolo26m_ft1_v3_1280/`, run record
  `experiments/20261123_det_yolo26m_ft1_v3_1280/`.
- Why not Explorer: the first Explorer run failed (PyPI torch 2.15 = CUDA 13, node driver 570 supports
  CUDA <= 12.8), and the E1 comparison only needs all trackers to share one documented detection
  file, not a specific machine. fp32 is at least as precise as fp16.
- Reproducible: a dev run with torch 2.15.0 (`af77fb6`) matches box for box (IoU >= 0.99 for all
  but 2 of 1.12 M boxes, max 0.66 px, score diff <= 1.3e-5) and gives identical recall/precision.
- Explorer jobs with the CUDA 12.6 build remain an optional cross-check; they do not replace this run.

## Environment
torch 2.14.1 / torchvision 0.29.1 pinned for all machines; on Linux uv installs the CUDA 12.6 build
(`[tool.uv.sources]` in `pyproject.toml`), which runs on driver 570 and on V100/A100/H200.

## NMS happens before clipping (standard Ultralytics output, kept)
Ultralytics 8.4.216 `DetectionPredictor`: `non_max_suppression` (torchvision NMS, IoU 0.7) runs
on letterboxed, unclipped boxes; `construct_result` → `ops.scale_boxes` then rescales and
`clip_boxes` clamps to the frame. Clipping cuts off the non-overlapping outside parts of two boxes
at the border, so their IoU can rise above 0.7 after NMS.
- Measured (`scripts/tools/nms_edge_check.py`): 392 of 87.2 M same-frame pairs exceed 0.7 (max
  0.798, MOT20-03); **all 392 touch the image edge, 0 in the interior**.
- Decision: keep the standard output (no second NMS after clipping): the effect is 0.03 % of boxes,
  only at the border where people enter/exit, and our detections stay comparable to normal YOLO use.

## detections-1.0
`video_id, frame (int32, 1-based), det_id (int32, 0..n-1 by descending score), x1, y1, x2, y2,
score (float32)` - defined in `src/odins_eye/io/schemas.py`. `det_id` order can swap between runs
for near-tied scores; compare runs by IoU matching (`scripts/tools/compare_detections.py`).

## GPU independence
Same code on H200, A100, V100, MPS, CPU (`device: auto`). Numerics differ slightly between devices
and library versions (sub-pixel boxes, ~1e-5 scores), so each run stores device and versions in
`env.json`; tracker comparisons are unaffected because they share one detection file.

## Not decided here
Fine-tuning on MOT20 (would require leave-one-sequence-out evaluation); NMS IoU ablation.
