"""Compare two detections-1.0 runs (e.g. official vs dev) row by row and by IoU matching.

Usage: python scripts/tools/compare_detections.py <dir_a> <dir_b> [out_csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from odins_eye.io.schemas import DETECTIONS_SCHEMA, read_table

SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
BOX = ["x1", "y1", "x2", "y2"]


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    same_rows = (
        len(a) == len(b) and (a[["frame", "det_id"]].values == b[["frame", "det_id"]].values).all()
    )
    out = {"rows_a": len(a), "rows_b": len(b), "same_frame_detid_order": bool(same_rows)}
    if same_rows:
        out["max_abs_box_diff_px"] = float(np.abs(a[BOX].values - b[BOX].values).max())
        out["max_abs_score_diff"] = float(np.abs(a["score"].values - b["score"].values).max())
    # order-independent check: match boxes per frame at IoU >= 0.99
    bf = dict(tuple(b.groupby("frame")))
    matched = unmatched = 0
    for frame, da in a.groupby("frame"):
        db = bf.get(frame)
        if db is None:
            unmatched += len(da)
            continue
        m = iou(da[BOX].to_numpy(float), db[BOX].to_numpy(float))
        r, c = linear_sum_assignment(-m)
        ok = m[r, c] >= 0.99
        matched += int(ok.sum())
        unmatched += len(da) - int(ok.sum()) + (len(db) - int(ok.sum()))
    out["boxes_matched_iou>=0.99"] = matched
    out["boxes_unmatched"] = unmatched
    return out


def main(dir_a: str, dir_b: str, out_csv: str | None = None) -> None:
    rows = []
    for seq in SEQS:
        a = read_table(Path(dir_a) / f"{seq}.parquet", DETECTIONS_SCHEMA)
        b = read_table(Path(dir_b) / f"{seq}.parquet", DETECTIONS_SCHEMA)
        rows.append({"seq": seq, **compare(a, b)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if out_csv:
        df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
