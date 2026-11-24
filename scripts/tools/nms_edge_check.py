"""Check NMS output: count same-frame box pairs with IoU above the NMS threshold, split by
whether a box touches the image edge (Ultralytics clips boxes after NMS, see ADR-003).

Usage: python scripts/tools/nms_edge_check.py <detections_dir> [iou_thr=0.7]
"""

import sys
from pathlib import Path

import numpy as np

from odins_eye.io.schemas import DETECTIONS_SCHEMA, read_table
from odins_eye.io.seqinfo import read_seqinfo
from odins_eye.utils.paths import DATA_DIR

SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def main(det_dir: str, thr: float = 0.7) -> None:
    for seq in SEQS:
        d = read_table(Path(det_dir) / f"{seq}.parquet", DETECTIONS_SCHEMA)
        info = read_seqinfo(DATA_DIR / "raw" / "MOT20" / "train" / seq)
        w, h = info.im_width, info.im_height
        above = at_edge = 0
        for _, g in d.groupby("frame"):
            b = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            ix = np.clip(
                np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0]),
                0,
                None,
            )
            iy = np.clip(
                np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1]),
                0,
                None,
            )
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iou = ix * iy / (area[:, None] + area[None, :] - ix * iy)
            edge = (b[:, 0] <= 0.5) | (b[:, 1] <= 0.5) | (b[:, 2] >= w - 0.5) | (b[:, 3] >= h - 0.5)
            r, c = np.where(np.triu(iou, 1) > thr + 1e-6)
            above += len(r)
            at_edge += int((edge[r] | edge[c]).sum())
        interior = above - at_edge
        print(f"{seq}: pairs IoU>{thr}: {above}, touching edge: {at_edge}, interior: {interior}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.7)
