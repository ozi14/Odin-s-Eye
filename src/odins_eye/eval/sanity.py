"""Known-answer tests for the evaluator: perturb ground truth so the correct score is known.

If the evaluator (or our MOT writer) is wrong, at least one of these cases disagrees with the
number derived by hand. This is E0's "calibrate the ruler" step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from odins_eye.eval.trackeval_runner import evaluate_tracks
from odins_eye.io.schemas import TRACKS_SCHEMA

# MOTChallenge distractor classes (TrackEval): person_on_vehicle 2, static_person 7,
# distractor 8, reflection 12; MOT20 also non_mot_vehicle 6.
DISTRACTOR_CLASSES = {"MOT20": (2, 6, 7, 8, 12), "MOT17": (2, 7, 8, 12)}
N_ALPHAS = 19  # HOTA IoU thresholds 0.05, 0.10, ..., 0.95


def scored_gt(gt: pd.DataFrame) -> pd.DataFrame:
    """The GT rows TrackEval scores (consider=1, pedestrian), as a tracks-1.0 table."""
    return gt.loc[gt["consider"] & (gt["mot_class"] == 1), TRACKS_SCHEMA.names].reset_index(
        drop=True
    )


def drop_rows(tracks: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    keep = np.random.default_rng(seed).random(len(tracks)) >= frac
    return tracks[keep].reset_index(drop=True)


def shift_x(tracks: pd.DataFrame, frac_of_width: float) -> pd.DataFrame:
    """Shift every box right by a fraction of its width: IoU = (1 - f) / (1 + f) for all boxes."""
    out = tracks.copy()
    dx = frac_of_width * (out["x2"] - out["x1"])
    out["x1"] += dx
    out["x2"] += dx
    return out


def swap_ids(tracks: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    """Swap the IDs of the two longest tracks from a frame where both exist (and did before)."""
    longest = tracks["track_id"].value_counts().index[:2]
    a, b = int(longest[0]), int(longest[1])
    fa = set(tracks.loc[tracks["track_id"] == a, "frame"])
    fb = set(tracks.loc[tracks["track_id"] == b, "frame"])
    both = sorted(f for f in fa & fb if f - 1 in fa and f - 1 in fb)
    if not both:
        raise ValueError("the two longest tracks never overlap in consecutive frames")
    f0 = both[len(both) // 2]
    out = tracks.copy()
    late = out["frame"] >= f0
    out.loc[late & (tracks["track_id"] == a), "track_id"] = b
    out.loc[late & (tracks["track_id"] == b), "track_id"] = a
    return out, (a, b, f0)


def add_distractor_boxes(
    tracks: pd.DataFrame, gt: pd.DataFrame, benchmark: str = "MOT20"
) -> tuple[pd.DataFrame, int]:
    """Add exact predictions on every distractor (static person, reflection, ...)."""
    extra = gt.loc[gt["mot_class"].isin(DISTRACTOR_CLASSES[benchmark]), TRACKS_SCHEMA.names].copy()
    extra["track_id"] += int(tracks["track_id"].max()) + 1  # new IDs, no clash
    return pd.concat([tracks, extra], ignore_index=True), len(extra)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    iy = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(ix, 0, None) * np.clip(iy, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def distractor_collisions(tracks: pd.DataFrame, gt: pd.DataFrame, benchmark: str = "MOT20") -> int:
    """Count predictions the MOTChallenge preprocessing will remove as distractor matches.

    Independent re-implementation of the rule: per frame, Hungarian-match predictions to ALL GT
    boxes (IoU >= 0.5); predictions matched to a distractor class are dropped. In a dense crowd a
    shifted box can slide onto a neighbouring static person and be removed, becoming a FN.
    """
    boxes = ["x1", "y1", "x2", "y2"]
    distractors = np.array(DISTRACTOR_CLASSES[benchmark])
    removed = 0
    gt_by_frame = dict(tuple(gt.groupby("frame")))
    for frame, pred in tracks.groupby("frame"):
        g = gt_by_frame.get(frame)
        if g is None:
            continue
        sim = _iou(pred[boxes].to_numpy(float), g[boxes].to_numpy(float))
        sim[sim < 0.5 - np.finfo(float).eps] = 0
        rows, cols = linear_sum_assignment(-sim)
        matched = sim[rows, cols] > np.finfo(float).eps
        removed += int(np.isin(g["mot_class"].to_numpy()[cols[matched]], distractors).sum())
    return removed


def known_answer_suite(
    gt: pd.DataFrame,
    gt_folder: str | Path,
    work_dir: str | Path,
    benchmark: str = "MOT20",
    drop_frac: float = 0.10,
    shift_frac: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Run 5 perturbations of one sequence's GT; return case | check | expected | got | ok."""
    seq = str(gt["video_id"].iloc[0])
    base = scored_gt(gt)
    n = len(base)
    work_dir = Path(work_dir)

    def score(name: str, tracks: pd.DataFrame) -> pd.Series:
        df = evaluate_tracks({seq: tracks}, gt_folder, work_dir / name, name, benchmark)
        return df.loc[seq]

    rows: list[dict] = []

    def check(case: str, metric: str, expected: float, got: float, tol: float = 0.01) -> None:
        rows.append(
            {
                "seq": seq,
                "case": case,
                "check": metric,
                "expected": round(float(expected), 4),
                "got": round(float(got), 4),
                "ok": bool(abs(float(got) - float(expected)) <= tol),
            }
        )

    r = score("gt_copy", base)
    for m in ("HOTA", "DetA", "AssA", "MOTA", "IDF1"):
        check("gt_copy", m, 100, r[m])
    for m in ("FP", "FN", "IDSW"):
        check("gt_copy", m, 0, r[m])

    tracks, n_extra = add_distractor_boxes(base, gt, benchmark)
    r = score("distractors", tracks)
    check(f"distractors (+{n_extra} boxes)", "HOTA", 100, r["HOTA"])
    check(f"distractors (+{n_extra} boxes)", "FP", 0, r["FP"])

    tracks = drop_rows(base, drop_frac, seed)
    k = n - len(tracks)
    r = score("drop", tracks)
    check(f"drop {drop_frac:.0%} of boxes", "FN", k, r["FN"])
    check(f"drop {drop_frac:.0%} of boxes", "FP", 0, r["FP"])
    check(f"drop {drop_frac:.0%} of boxes", "DetA", 100 * (n - k) / n, r["DetA"])
    check(f"drop {drop_frac:.0%} of boxes", "MOTA", 100 * (n - k) / n, r["MOTA"])
    check(f"drop {drop_frac:.0%} of boxes", "IDSW", 0, r["IDSW"])

    tracks, _ = swap_ids(base)
    r = score("swap", tracks)
    check("swap 2 IDs once", "IDSW", 2, r["IDSW"])
    check("swap 2 IDs once", "DetA", 100, r["DetA"])
    check("swap 2 IDs once", "MOTA", 100 * (n - 2) / n, r["MOTA"])
    rows.append({**rows[-1], "check": "AssA < 100", "expected": 1, "got": int(r["AssA"] < 100)})
    rows[-1]["ok"] = bool(r["AssA"] < 100)

    iou = (1 - shift_frac) / (1 + shift_frac)
    passing = sum(1 for k_ in range(1, N_ALPHAS + 1) if iou >= 0.05 * k_ - 1e-9)
    shifted = shift_x(base, shift_frac)
    lost = distractor_collisions(shifted, gt, benchmark)  # shifted onto a static person etc.
    r = score("shift", shifted)
    case = f"shift {shift_frac:.0%} of width (IoU {iou:.3f})"
    check(case, "FN (distractor collisions)", lost, r["FN"])
    check(case, "MOTA", 100 * (n - lost) / n, r["MOTA"])  # IoU > 0.5: CLEAR sees ~perfect
    check(case, "HOTA", 100 * passing / N_ALPHAS * (n - lost) / n, r["HOTA"], tol=0.05)
    check(case, "MOTP", 100 * iou, r["MOTP"], tol=0.05)
    return pd.DataFrame(rows)
