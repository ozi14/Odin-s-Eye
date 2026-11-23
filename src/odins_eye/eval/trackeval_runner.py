"""Thin wrapper around TrackEval: the official MOTChallenge metrics (HOTA, CLEAR, Identity).

Validated in E0 (scripts/91_e0_validate_eval.py, ADR-002). Rules it enforces:
- BENCHMARK must match the ground truth: for MOT20, TrackEval adds `non_mot_vehicle` to the
  distractor classes.
- Distractor preprocessing stays on: predictions on static people, reflections, etc. are removed,
  not counted as false positives (the v3 bug).
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from odins_eye.io.mot import write_mot_results

RATIO_METRICS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDP", "IDR"]  # in %
COUNT_METRICS = ["IDSW", "Frag", "TP", "FP", "FN", "Dets", "GT_Dets", "IDs", "GT_IDs"]
COLUMNS = RATIO_METRICS + COUNT_METRICS
COMBINED = "COMBINED"


def _row(r: dict) -> dict:
    h, c, i, n = r["HOTA"], r["CLEAR"], r["Identity"], r["Count"]
    return {
        "HOTA": 100 * float(h["HOTA"].mean()),  # HOTA, DetA, AssA, LocA: mean over 19 IoU alphas
        "DetA": 100 * float(h["DetA"].mean()),
        "AssA": 100 * float(h["AssA"].mean()),
        "LocA": 100 * float(h["LocA"].mean()),
        "MOTA": 100 * float(c["MOTA"]),  # CLEAR metrics: single IoU threshold 0.5
        "MOTP": 100 * float(c["MOTP"]),
        "IDF1": 100 * float(i["IDF1"]),
        "IDP": 100 * float(i["IDP"]),
        "IDR": 100 * float(i["IDR"]),
        "IDSW": int(c["IDSW"]),
        "Frag": int(c["Frag"]),
        "TP": int(c["CLR_TP"]),
        "FP": int(c["CLR_FP"]),
        "FN": int(c["CLR_FN"]),
        "Dets": int(n["Dets"]),
        "GT_Dets": int(n["GT_Dets"]),
        "IDs": int(n["IDs"]),
        "GT_IDs": int(n["GT_IDs"]),
    }


def mot_challenge_dataset(
    trackers_folder: str | Path,
    tracker_name: str,
    gt_folder: str | Path,
    seqs: Sequence[str],
    benchmark: str = "MOT20",
):
    """Open MOT predictions and GT with the shared TrackEval preprocessing settings."""
    import trackeval  # lazy: E0 swaps in the upstream reference implementation

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "BENCHMARK": benchmark,
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {seq: None for seq in seqs},  # None: length read from seqinfo.ini
            "CLASSES_TO_EVAL": ["pedestrian"],
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        }
    )
    return trackeval.datasets.MotChallenge2DBox(ds_cfg)


def evaluate_mot_files(
    trackers_folder: str | Path,
    tracker_name: str,
    gt_folder: str | Path,
    seqs: Sequence[str],
    benchmark: str = "MOT20",
) -> pd.DataFrame:
    """Score `<trackers_folder>/<tracker>/data/<seq>.txt` against `<gt_folder>/<seq>/gt/gt.txt`.

    Returns one row per sequence plus a COMBINED row (TrackEval's own combination, not a mean).
    """
    import trackeval  # lazy: E0 swaps in the upstream reference implementation

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update(
        {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):  # TrackEval prints configs and progress
        dataset = mot_challenge_dataset(trackers_folder, tracker_name, gt_folder, seqs, benchmark)
        metrics = [
            trackeval.metrics.HOTA(),
            trackeval.metrics.CLEAR(),
            trackeval.metrics.Identity(),
        ]
        results, _ = trackeval.Evaluator(eval_cfg).evaluate([dataset], metrics)
    per_tracker = results["MotChallenge2DBox"][tracker_name]
    rows = {seq: _row(per_tracker[seq]["pedestrian"]) for seq in seqs}
    rows[COMBINED] = _row(per_tracker["COMBINED_SEQ"]["pedestrian"])
    return pd.DataFrame.from_dict(rows, orient="index")[COLUMNS].rename_axis("seq")


def evaluate_tracks(
    tracks_by_seq: Mapping[str, pd.DataFrame],
    gt_folder: str | Path,
    work_dir: str | Path,
    tracker_name: str = "tracker",
    benchmark: str = "MOT20",
) -> pd.DataFrame:
    """Write tracks-1.0 tables as MOT files under `work_dir`, then score them."""
    work_dir = Path(work_dir)
    for seq, tracks in tracks_by_seq.items():
        write_mot_results(tracks, work_dir / tracker_name / "data" / f"{seq}.txt")
    return evaluate_mot_files(work_dir, tracker_name, gt_folder, list(tracks_by_seq), benchmark)
