"""
03_benchmark.py — Phase 3: MOT Metrics Evaluation

Evaluates Phase 1 tracking output against MOT20 ground truth using the
standard MOTChallenge metrics:

  MOTA  — Multi-Object Tracking Accuracy  (penalises FP, FN, ID switches)
  MOTP  — Multi-Object Tracking Precision (avg localisation error)
  IDF1  — ID F1 score                     (identity consistency)
  HOTA  — Higher-Order Tracking Accuracy  (if motmetrics ≥ 1.3)
  MT    — Mostly Tracked (>80% frames)
  ML    — Mostly Lost (<20% frames)

Uses the `motmetrics` library (pip install motmetrics).

Compares three ablations:
  1. ByteTrack baseline (IoU only)
  2. ByteTrack + DINOv2 ReID
  3. Full pipeline (with VLM narration, qualitative only)

Usage:
    cd mot
    python scripts/03_benchmark.py --seq MOT20-01
    python scripts/03_benchmark.py --seq MOT20-01 MOT20-02 MOT20-03 MOT20-05
    python scripts/03_benchmark.py --seq MOT20-01 --pred_suffix _no_reid
"""

import os
import sys
import argparse
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
logger = logging.getLogger(__name__)

# ── NumPy 2.0 compatibility shim for motmetrics ─────────────────────
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────
# MOTChallenge gt.txt / prediction txt loader
# ─────────────────────────────────────────────────────────────────────

def load_mot_txt(path: Path) -> dict:
    """
    Load a MOTChallenge-format txt file.

    Returns dict: {frame_id: [(track_id, x, y, w, h, conf), ...]}
    Each row format: frame, id, x, y, w, h, conf, cx, cy, cz
    """
    data = {}
    if not path.exists():
        logger.warning("Prediction file not found: %s", path)
        return data
    lines = path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        try:
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
        except ValueError:
            logger.warning("Skipping malformed MOT row: %s", line)
            continue
        if conf < 0:   # MOT gt uses -1 for ignored regions
            continue
        data.setdefault(fid, []).append((tid, x, y, w, h))
    return data


def load_gt_txt(path: Path) -> dict:
    """
    Load MOT20 gt.txt.  Rows with class != 1 or conf = 0 are ignored.

    GT format: frame, id, x, y, w, h, conf, class, visibility
    We keep conf > 0 and class == 1 (pedestrian).
    """
    data = {}
    if not path.exists():
        logger.warning("GT file not found: %s", path)
        return data
    lines = path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 7:
            continue
        try:
            fid = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = int(parts[6])
            cls = int(parts[7]) if len(parts) > 7 else 1
        except ValueError:
            logger.warning("Skipping malformed GT row: %s", line)
            continue
        if conf == 0 or cls != 1:
            continue
        data.setdefault(fid, []).append((tid, x, y, w, h))
    return data


# ─────────────────────────────────────────────────────────────────────
# motmetrics evaluation
# ─────────────────────────────────────────────────────────────────────

def evaluate_mot(gt_data: dict, pred_data: dict,
                 iou_thresh: float = 0.5) -> dict:
    """
    Compute MOTA/MOTP/IDF1/etc. using the `motmetrics` library.

    Args:
        gt_data:   {frame_id: [(tid, x, y, w, h), ...]}
        pred_data: same format
        iou_thresh: IoU threshold for a successful match

    Returns dict of metric values.
    """
    try:
        import motmetrics as mm
    except ImportError:
        logger.error("motmetrics not installed. Run: pip install motmetrics")
        return {}

    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt_data) | set(pred_data))
    for fid in all_frames:
        gt_objs  = gt_data.get(fid, [])
        pr_objs  = pred_data.get(fid, [])

        gt_ids   = [o[0] for o in gt_objs]
        pr_ids   = [o[0] for o in pr_objs]

        # motmetrics expects [x, y, w, h] (top-left + size), NOT xyxy
        gt_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in gt_objs],
                            dtype=np.float32) if gt_objs else np.empty((0, 4))
        pr_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in pr_objs],
                            dtype=np.float32) if pr_objs else np.empty((0, 4))

        if len(gt_boxes) and len(pr_boxes):
            dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=1.0 - iou_thresh)
        else:
            dist = mm.distances.iou_matrix(
                np.empty((len(gt_objs), 4)) if not len(gt_boxes) else gt_boxes,
                np.empty((len(pr_objs), 4)) if not len(pr_boxes) else pr_boxes,
                max_iou=1.0 - iou_thresh,
            )

        acc.update(gt_ids, pr_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'num_frames', 'mota', 'motp', 'idf1',
        'num_switches', 'num_fragmentations',
        'precision', 'recall',
        'mostly_tracked', 'mostly_lost', 'partially_tracked',
        'num_false_positives', 'num_misses',
        'num_unique_objects',
    ], name='seq')

    result = {}
    for col in summary.columns:
        val = summary[col].iloc[0]
        result[col] = round(float(val), 4) if not np.isnan(float(val)) else 0.0
    return result


# ─────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────

def print_metrics(seq_name: str, metrics: dict):
    w = 60
    print(f"\n{'='*w}")
    print(f"  {seq_name}")
    print(f"{'='*w}")
    if not metrics:
        print("  No metrics computed.")
        return

    key_metrics = [
        ('MOTA',   'mota'),
        ('MOTP',   'motp'),
        ('IDF1',   'idf1'),
        ('ID Sw.', 'num_switches'),
        ('MT',     'mostly_tracked'),
        ('ML',     'mostly_lost'),
        ('Prec',   'precision'),
        ('Rec',    'recall'),
        ('FP',     'num_false_positives'),
        ('FN',     'num_misses'),
        ('Frags',  'num_fragmentations'),
        ('GT IDs', 'num_unique_objects'),
    ]
    for label, key in key_metrics:
        val = metrics.get(key, 'N/A')
        if isinstance(val, float):
            if key in ('mota', 'motp', 'idf1', 'precision', 'recall'):
                print(f"  {label:<12}  {val*100:6.2f} %")
            else:
                print(f"  {label:<12}  {val:6.1f}")
        else:
            print(f"  {label:<12}  {val}")
    print(f"{'='*w}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 3: MOT benchmark evaluation')
    p.add_argument('--seq', nargs='+', required=True,
                   help='Sequence(s) to evaluate, e.g. MOT20-01')
    p.add_argument('--dataset_dir', default=str(BASE_DIR / 'datasets' / 'MOT20'))
    p.add_argument('--output_dir', default=str(BASE_DIR / 'output'))
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--pred_suffix', default='',
                   help='Suffix for mot_results filename, e.g. _no_reid')
    p.add_argument('--iou_thresh', type=float, default=0.5,
                   help='IoU threshold for GT-prediction matching (default: 0.5)')
    p.add_argument('--save_json', action='store_true',
                   help='Save metrics as JSON alongside mot_results.txt')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)

    all_metrics = {}
    for seq_name in args.seq:
        gt_path   = dataset_dir / args.split / seq_name / 'gt' / 'gt.txt'
        pred_path = output_dir / seq_name / f"mot_results{args.pred_suffix}.txt"

        if not gt_path.exists():
            logger.warning("GT not found for %s: %s", seq_name, gt_path)
            continue
        if not pred_path.exists():
            logger.warning("Predictions not found for %s: %s", seq_name, pred_path)
            continue

        logger.info("Loading sequence: %s", seq_name)
        logger.info("  GT: %s", gt_path)
        logger.info("  Pred: %s", pred_path)

        gt_data   = load_gt_txt(gt_path)
        pred_data = load_mot_txt(pred_path)

        metrics = evaluate_mot(gt_data, pred_data, args.iou_thresh)
        print_metrics(seq_name, metrics)
        all_metrics[seq_name] = metrics

        if args.save_json:
            import json
            json_path = output_dir / seq_name / f"metrics{args.pred_suffix}.json"
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info("Metrics saved: %s", json_path)

    # Summary across all sequences
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY  (all sequences)")
        print(f"{'='*60}")
        for seq, m in all_metrics.items():
            mota = m.get('mota', 0) * 100
            idf1 = m.get('idf1', 0) * 100
            sw   = int(m.get('num_switches', 0))
            print(f"  {seq:<14}  MOTA {mota:5.2f}%  IDF1 {idf1:5.2f}%  "
                  f"ID Sw. {sw:4d}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
