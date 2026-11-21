"""Step 1: convert MOT20 ground truth into gt_tracks-1.0 Parquet and print a dataset summary.

Usage:  python scripts/00_prepare_data.py            (defaults from configs/)
        python scripts/00_prepare_data.py data.sequences=[MOT20-01]
"""

from __future__ import annotations

import hydra
import pandas as pd
from omegaconf import DictConfig

from odins_eye.io.mot import read_mot_gt
from odins_eye.io.schemas import GT_TRACKS_SCHEMA, write_table
from odins_eye.io.seqinfo import read_seqinfo
from odins_eye.utils.paths import PROJECT_ROOT


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raw_dir = PROJECT_ROOT / cfg.data.raw_dir / cfg.data.split
    out_dir = PROJECT_ROOT / cfg.data.gt_tracks_dir
    rows = []
    for seq in cfg.data.sequences:
        info = read_seqinfo(raw_dir / seq)
        gt = read_mot_gt(raw_dir / seq, anchor=cfg.data.anchor)
        write_table(gt, out_dir / f"{seq}.parquet", GT_TRACKS_SCHEMA)

        ped = gt[gt["consider"] & (gt["mot_class"] == 1)]  # what TrackEval scores
        per_frame = ped.groupby("frame").size().reindex(range(1, info.seq_length + 1), fill_value=0)
        rows.append(
            {
                "seq": seq,
                "size": f"{info.im_width}x{info.im_height}",
                "fps": info.frame_rate,
                "frames": info.seq_length,
                "gt_rows": len(gt),
                "ped_ids": ped["track_id"].nunique(),
                "ped/frame mean": round(per_frame.mean(), 1),
                "ped/frame max": int(per_frame.max()),
                "static rows (cls 7)": int((gt["mot_class"] == 7).sum()),
                "ped vis<0.25": f"{(ped['visibility'] < 0.25).mean():.1%}",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {len(rows)} files to {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
