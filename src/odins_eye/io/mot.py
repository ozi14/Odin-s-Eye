"""MOTChallenge text files <-> our Parquet tables (tracks-1.0 / gt_tracks-1.0).

MOT text uses (bb_left, bb_top, bb_width, bb_height); our tables use corners (x1, y1, x2, y2).
The conversion happens only here, at the file boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from odins_eye.io.schemas import GT_TRACKS_SCHEMA, TRACKS_SCHEMA, to_table
from odins_eye.io.seqinfo import read_seqinfo

_GT_COLS = ["frame", "id", "left", "top", "width", "height", "consider", "mot_class", "visibility"]


def _corners(raw: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "x1": raw["left"],
        "y1": raw["top"],
        "x2": raw["left"] + raw["width"],
        "y2": raw["top"] + raw["height"],
    }


def read_mot_gt(seq_dir: str | Path, anchor: str = "body") -> pd.DataFrame:
    """Read `gt/gt.txt` of one sequence, unfiltered (all classes), as gt_tracks-1.0."""
    seq_dir = Path(seq_dir)
    info = read_seqinfo(seq_dir)
    raw = pd.read_csv(seq_dir / "gt" / "gt.txt", header=None)
    if raw.shape[1] != len(_GT_COLS):
        raise ValueError(f"{seq_dir}/gt/gt.txt: expected 9 columns, got {raw.shape[1]}")
    raw.columns = _GT_COLS
    df = pd.DataFrame(
        {
            "video_id": info.name,
            "frame": raw["frame"],
            "track_id": raw["id"],
            **_corners(raw),
            "score": 1.0,
            "anchor": anchor,
            "consider": raw["consider"].astype(bool),
            "mot_class": raw["mot_class"],
            "visibility": raw["visibility"],
        }
    )
    df = df.sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True)
    return to_table(df, GT_TRACKS_SCHEMA).to_pandas()


def read_mot_results(path: str | Path, video_id: str, anchor: str = "body") -> pd.DataFrame:
    """Read a tracker result file (frame,id,x,y,w,h,conf,...) as tracks-1.0."""
    path = Path(path)
    if path.stat().st_size == 0:
        return to_table(pd.DataFrame(columns=TRACKS_SCHEMA.names), TRACKS_SCHEMA).to_pandas()
    raw = pd.read_csv(path, header=None).iloc[:, :7]
    raw.columns = ["frame", "id", "left", "top", "width", "height", "score"]
    df = pd.DataFrame(
        {
            "video_id": video_id,
            "frame": raw["frame"],
            "track_id": raw["id"],
            **_corners(raw),
            "score": raw["score"],
            "anchor": anchor,
        }
    )
    return to_table(df, TRACKS_SCHEMA).to_pandas()


def write_mot_results(tracks: pd.DataFrame, path: str | Path) -> Path:
    """Write one video's tracks in MOTChallenge format for TrackEval."""
    table = to_table(tracks, TRACKS_SCHEMA).to_pandas()  # validates; drops extra columns
    if table["video_id"].nunique() > 1:
        raise ValueError("write_mot_results expects tracks of exactly one video")
    table = table.sort_values(["frame", "track_id"], kind="stable")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in table.itertuples(index=False):
            w, h = r.x2 - r.x1, r.y2 - r.y1
            f.write(
                f"{r.frame},{r.track_id},{r.x1:.2f},{r.y1:.2f},{w:.2f},{h:.2f},"
                f"{r.score:.4f},-1,-1,-1\n"
            )
    return path
