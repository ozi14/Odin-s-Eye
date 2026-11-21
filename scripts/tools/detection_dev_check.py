"""Quick detection check against MOT20 GT (development aid, not an official metric).

Per sequence: recall/precision at conf 0.1/0.3/0.5 (IoU >= 0.5, Hungarian per frame) and recall by
visibility. Boxes matched to distractor GT (static person, ...) count as neither TP nor FP, like
TrackEval. Official detection numbers come from the validated TrackEval path.

Usage: python scripts/tools/detection_dev_check.py <detections_dir> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from odins_eye.io.schemas import DETECTIONS_SCHEMA, GT_TRACKS_SCHEMA, read_table
from odins_eye.utils.paths import DATA_DIR

SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
DISTRACTORS = (2, 6, 7, 8, 12)
BOX = ["x1", "y1", "x2", "y2"]
VIS_BINS = [0, 0.25, 0.5, 0.75, 1.01]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def check(dets: pd.DataFrame, gt: pd.DataFrame) -> dict:
    gt = gt[(gt["consider"] & (gt["mot_class"] == 1)) | gt["mot_class"].isin(DISTRACTORS)]
    by_frame = dict(tuple(gt.groupby("frame")))
    n_ped = int((gt["mot_class"] == 1).sum())
    out = {}
    for th in (0.1, 0.3, 0.5):
        tp = fp = 0
        vis_hit = []
        for frame, d in dets[dets["score"] >= th].groupby("frame"):
            g = by_frame.get(frame)
            if g is None:
                fp += len(d)
                continue
            m = iou(d[BOX].to_numpy(float), g[BOX].to_numpy(float))
            m[m < 0.5] = 0
            r, c = linear_sum_assignment(-m)
            ok = m[r, c] > 0
            c = c[ok]
            ped = g["mot_class"].to_numpy()[c] == 1
            tp += int(ped.sum())
            fp += len(d) - int(ok.sum())
            vis_hit += list(g["visibility"].to_numpy()[c[ped]])
        out[f"recall@{th}"] = tp / n_ped
        out[f"precision@{th}"] = tp / max(tp + fp, 1)
        if th == 0.1:
            allc = np.histogram(gt.loc[gt["mot_class"] == 1, "visibility"], VIS_BINS)[0]
            hitc = np.histogram(vis_hit, VIS_BINS)[0]
            for lo, hi, h, a in zip(
                ["0", "25", "50", "75"], ["25", "50", "75", "100"], hitc, allc, strict=True
            ):
                out[f"recall_vis_{lo}-{hi}"] = h / max(a, 1)
    out["gt_ped_per_frame"] = n_ped / gt["frame"].nunique()
    out["det_per_frame"] = len(dets) / dets["frame"].nunique()
    return out


def main(det_dir: str, out_dir: str) -> None:
    rows = []
    for seq in SEQS:
        dets = read_table(Path(det_dir) / f"{seq}.parquet", DETECTIONS_SCHEMA)
        gt = read_table(
            DATA_DIR / "interim" / "gt_tracks" / "MOT20" / f"{seq}.parquet", GT_TRACKS_SCHEMA
        )
        rows.append({"seq": seq, **check(dets, gt)})
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(Path(out_dir) / "detection_check.csv", index=False)
    print(df.round(3).to_string(index=False))
    json.dump(
        {"detections": str(det_dir)}, open(Path(out_dir) / "detection_check_source.json", "w")
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
