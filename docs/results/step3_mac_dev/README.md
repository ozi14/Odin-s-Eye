# Step 3 results: detection on MOT20 (Mac development run)

> **Development results, not official.** Run on Taha's Mac (Apple M4 Max, MPS, fp32). The official
> Step 3 run is on Explorer (CUDA, fp16) and goes to `docs/results/step3_explorer/` when it finishes.
> Use these detections to develop Step 4 (tracking); report numbers only from the Explorer run.

## What was run

```bash
python scripts/10_detect.py experiment.desc=yolo26m_ft1_v3_1280_mac_dev
python scripts/tools/detection_dev_check.py data/interim/detections/yolo26m_ft1_v3_1280_mac_dev docs/results/step3_mac_dev
```

| Setting | Value |
|---|---|
| Code | commit `af77fb6` (clean), see `git.json` |
| Detector | `yolo26m_ft1_v3_1280.pt` (sha256 `379b9bbd…`), ADR-003 |
| imgsz / conf / NMS IoU / max_det | 1280 / 0.1 / 0.7 / 1000 |
| Device | MPS (Apple GPU), fp32, batch 16 (`env.json`) |
| Output | `data/interim/detections/yolo26m_ft1_v3_1280_mac_dev/MOT20-0X.parquet` (detections-1.0, ~1.12 M boxes) |

Ask Taha for the parquet files (29 MB) if you need them; they are not in git.

## Results

| Seq | GT ped / frame | Det / frame | Recall @0.1 | Precision @0.1 | Recall @0.3 | Recall @0.5 | fps (MPS) |
|---|---|---|---|---|---|---|---|
| MOT20-01 | 46.3 | 45.4 | **85.0%** | 95.1% | 81.5% | 78.3% | 32 |
| MOT20-02 | 55.6 | 52.5 | **84.0%** | 95.1% | 80.1% | 76.1% | 36 |
| MOT20-03 | 130.4 | 131.8 | **87.3%** | 91.6% | 82.9% | 78.4% | 30 |
| MOT20-05 | 195.0 | 192.0 | **81.2%** | 93.2% | 74.5% | 68.4% | 27 |

Recall by visibility (conf >= 0.1):

| Seq | 0–25% visible | 25–50% | 50–75% | 75–100% |
|---|---|---|---|---|
| MOT20-01 | 64% | 94% | 94% | 99% |
| MOT20-02 | 65% | 95% | 98% | 98% |
| MOT20-03 | 53% | 84% | 94% | 94% |
| MOT20-05 | 52% | 80% | 91% | 93% |

Matching: IoU >= 0.5, Hungarian per frame, against scored GT pedestrians (`consider=1`, class 1).
Boxes on distractors (static people, reflections, ...) count as neither hits nor errors, as in
TrackEval. Full table: `detection_check.csv`.

## Findings

1. **conf 0.1 is the right threshold.** 0.5 -> 0.1 adds 7–13 recall points for 3–5 precision points;
   the extra boxes are mostly partly hidden people, which ByteTrack's second pass uses.
2. **Occlusion is the main failure** (input for Q3): 93–99% recall for visible people, 52–65% for
   people less than 25% visible.
3. **Denser is harder:** MOT20-05 (densest) has the lowest recall, 81% (68% at conf 0.5).
4. **No truncation:** the densest frame has 232 boxes, far below `max_det` 1000.

## Files

| File | What |
|---|---|
| `step3_mac_dev_results.html` | interactive charts (open in a browser): people per frame over time, recall by visibility, confidence trade-off, score histogram |
| `frame_MOT20-05_1819.jpg` | densest frame: green conf >= 0.3, orange 0.1–0.3, red = missed GT pedestrian |
| `detection_check.csv` | all numbers above |
| `timing.csv`, `summary.json` | per-sequence detections and speed |
| `config.yaml`, `env.json`, `git.json` | exact config, hardware, code version |
