# Step 3 results: detection on MOT20 (official)

> **Official Step 3 output**, run on Taha's Mac (Apple M4 Max, MPS, fp32) at commit `41156db`
> (clean). A Mac dev run three days earlier (torch 2.15) gave the same result (see "Reproducibility").
> Decision and settings: [ADR-003](../../decisions/ADR-003-detector.md). Step 4 trackers read these files.

## What was run

```bash
python scripts/10_detect.py
python scripts/tools/detection_dev_check.py data/interim/detections/yolo26m_ft1_v3_1280 docs/results/step3
```

| Setting | Value |
|---|---|
| Code | commit `41156db` (clean), torch 2.14.1, ultralytics 8.4.216, see `git.json`, `env.json` |
| Detector | `yolo26m_ft1_v3_1280.pt` (sha256 `379b9bbd…`) |
| imgsz / conf / NMS IoU / max_det | 1280 / 0.1 / 0.7 / 1000 |
| Device | MPS (Apple GPU), fp32, batch 16 |
| Output | `data/interim/detections/yolo26m_ft1_v3_1280/MOT20-0X.parquet` (detections-1.0, 1,118,752 boxes) |

The parquet files (29 MB) are not in git; ask Taha for them or rerun the command above (~5 min on an M4 Max).

## Results

| Seq | GT ped / frame | Det / frame | Recall @0.1 | Precision @0.1 | Recall @0.3 | Recall @0.5 | fps (MPS) |
|---|---|---|---|---|---|---|---|
| MOT20-01 | 46.3 | 45.4 | **85.0%** | 95.1% | 81.5% | 78.3% | 29 |
| MOT20-02 | 55.6 | 52.5 | **84.0%** | 95.1% | 80.1% | 76.1% | 35 |
| MOT20-03 | 130.4 | 131.8 | **87.3%** | 91.6% | 82.9% | 78.4% | 30 |
| MOT20-05 | 195.0 | 192.0 | **81.2%** | 93.2% | 74.5% | 68.4% | 28 |

Recall by visibility (conf >= 0.1):

| Seq | 0–25% visible | 25–50% | 50–75% | 75–100% |
|---|---|---|---|---|
| MOT20-01 | 64% | 94% | 94% | 99% |
| MOT20-02 | 65% | 95% | 98% | 98% |
| MOT20-03 | 53% | 84% | 94% | 94% |
| MOT20-05 | 52% | 80% | 91% | 93% |

Matching: IoU >= 0.5, Hungarian per frame, against scored GT pedestrians (`consider=1`, class 1).
Boxes on distractors (static people, reflections, ...) count as neither hits nor errors, as in
TrackEval. Full table: `detection_check.csv`. This is a development check; tracking metrics come
from the validated TrackEval path (Step 2).

## Findings

1. **conf 0.1 is the right threshold.** 0.5 -> 0.1 adds 7–13 recall points for 3–5 precision points;
   14–25% of all boxes are in the 0.1–0.6 band that ByteTrack's second pass uses.
2. **Occlusion is the main failure** (input for Q3): 93–99% recall for visible people, 52–65% for
   people less than 25% visible.
3. **Denser is harder:** MOT20-05 (densest) has the lowest recall, 81% (68% at conf 0.5).
4. **No truncation:** the densest frame has 232 boxes, far below `max_det` 1000.
5. **Input sanity:** every frame has detections, no box outside the image, frame-to-frame box count
   changes at most 21.
6. **392 box pairs (0.03%) overlap above the NMS threshold, all at the image edge**, because
   Ultralytics clips boxes after NMS. Kept as standard output (ADR-003).

## Reproducibility (official vs Mac dev run)

Same machine, torch 2.14.1 (official) vs 2.15.0 (dev, commit `af77fb6`):

| Seq | Boxes (both) | Matched at IoU >= 0.99 | Max box diff (matched) | Max score diff |
|---|---|---|---|---|
| MOT20-01 | 19,458 | 19,458 | 0.57 px | 4e-6 |
| MOT20-02 | 146,026 | 146,026 | 0.0004 px | 4e-6 |
| MOT20-03 | 316,930 | 316,929 (1 box at IoU 0.977) | 0.66 px | 5e-6 |
| MOT20-05 | 636,338 | 636,337 (1 box at IoU 0.971) | 0.55 px | 1e-5 |

All recall/precision values are identical to 3 decimals. 38 near-tied score pairs swap `det_id`
order, so compare runs by IoU matching (`scripts/tools/compare_detections.py`), not row by row
(`official_vs_dev_rowwise.csv` shows why).

## Files

| File | What |
|---|---|
| `step3_mac_dev_results.html` | interactive charts (built from the dev run; identical numbers) |
| `frame_MOT20-05_1819.jpg` | densest frame: green conf >= 0.3, orange 0.1–0.3, red = missed GT pedestrian |
| `detection_check.csv` | all accuracy numbers above |
| `official_vs_dev_rowwise.csv` | row-by-row comparison with the dev run |
| `timing.csv`, `summary.json` | per-sequence detections and speed |
| `config.yaml`, `env.json`, `git.json` | exact config, hardware, code version |
