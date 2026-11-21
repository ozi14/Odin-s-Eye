# ADR-003: Detector for Phase 1 = yolo26m_ft1_v3_1280, frozen settings, detections-1.0

- Status: accepted · 2026-11-20 · owner: Taha
- Code: `src/odins_eye/perception/detector.py`, `scripts/10_detect.py`,
  config `configs/perception/detector/yolo26m_ft1_v3_1280.yaml`

## Decision
- Weights `yolo26m_ft1_v3_1280.pt` (sha256 379b9bbd…): YOLO26m fine-tuned on CrowdHuman + CCTV
  Persons (no MOT17/MOT20 frames), long Explorer run, epoch 160, imgsz 1280. Validation
  (CrowdHuman+CCTV val): P .882, R .850, mAP50 .922, mAP50-95 .667; best of all saved
  checkpoints (ft1_v1 .649, ft1_v2 .662).
- Settings: imgsz 1280 (= training), conf 0.1, NMS IoU 0.7 (the head is not end-to-end, so NMS
  runs), max_det 1000 (default 300 could truncate dense frames), fp16 on CUDA, class person only.
- Detection runs once per sequence; every tracker reads the same `detections-1.0` files.

## detections-1.0
`video_id, frame (int32, 1-based), det_id (int32, 0..n-1 by descending score), x1, y1, x2, y2,
score (float32), anchor` - defined in `src/odins_eye/io/schemas.py`.

## GPU independence
Same code on H200, A100, MPS, CPU (`device: auto`). Numerics can differ slightly between GPU
types, so each run stores the GPU in `env.json`; comparisons between trackers are unaffected
because they share one detection file.

## Not decided here
Fine-tuning on MOT20 (would require leave-one-sequence-out evaluation); NMS IoU ablation.
