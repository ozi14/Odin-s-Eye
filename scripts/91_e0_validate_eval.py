"""E0: validate the evaluation code before trusting any tracker number.

A) Reference check: score TrackEval's example tracker (MPNTrack, MOT20-train) with our wrapper
   (trackeval 1.3.0) and with the original upstream TrackEval; every metric must match.
B) Known answers: perturb the GT of each sequence (copy, distractors, drop, ID swap, shift) and
   compare against scores derived by hand.

Outputs: experiments/<date>_e0_trackeval-validation/
"""

from __future__ import annotations

import json
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from odins_eye.eval.sanity import known_answer_suite
from odins_eye.eval.trackeval_runner import evaluate_mot_files
from odins_eye.io.mot import read_mot_gt
from odins_eye.utils.paths import PROJECT_ROOT
from odins_eye.utils.run import new_run_dir


def reference_check(cfg: DictConfig, gt_folder, seqs) -> pd.DataFrame:
    ref = cfg.experiment.reference
    trackers_folder = PROJECT_ROOT / ref.trackers_folder
    ours = evaluate_mot_files(
        trackers_folder, ref.tracker, gt_folder, seqs, cfg.experiment.benchmark
    )
    upstream_dir = PROJECT_ROOT / ref.upstream_dir
    if not upstream_dir.exists():
        raise SystemExit(
            f"missing {upstream_dir}: git clone https://github.com/JonathonLuiten/TrackEval.git "
            "external/TrackEval && git -C external/TrackEval checkout 12c8791"
        )
    args = {
        "upstream_dir": str(upstream_dir),
        "trackers_folder": str(trackers_folder),
        "tracker": ref.tracker,
        "gt_folder": str(gt_folder),
        "seqs": list(seqs),
        "benchmark": cfg.experiment.benchmark,
    }
    script = PROJECT_ROOT / "scripts" / "tools" / "trackeval_reference.py"
    out = subprocess.run(
        [sys.executable, str(script), json.dumps(args)], capture_output=True, text=True, check=True
    )
    upstream = pd.read_json(out.stdout.strip().splitlines()[-1], orient="split")
    upstream.index.name = "seq"
    diff = (ours - upstream.loc[ours.index, ours.columns]).abs()
    return pd.concat({"ours": ours, "upstream": upstream[ours.columns], "abs_diff": diff}, axis=1)


@hydra.main(version_base=None, config_path="../configs", config_name="e0")
def main(cfg: DictConfig) -> None:
    run = new_run_dir(cfg.experiment.id, cfg.experiment.desc, cfg)
    gt_folder = PROJECT_ROOT / cfg.data.raw_dir / cfg.data.split
    seqs = list(cfg.data.sequences)

    ref = reference_check(cfg, gt_folder, seqs)
    ref.to_csv(run / "reference_check.csv")
    max_diff = float(np.nanmax(ref["abs_diff"].to_numpy()))
    ref_ok = max_diff < 1e-9

    ka = pd.concat(
        [
            known_answer_suite(
                read_mot_gt(gt_folder / seq, anchor=cfg.data.anchor),
                gt_folder,
                run / "work" / seq,
                cfg.experiment.benchmark,
                **cfg.experiment.known_answers,
            )
            for seq in seqs
        ],
        ignore_index=True,
    )
    ka.to_csv(run / "known_answers.csv", index=False)

    summary = {
        "reference_tracker": cfg.experiment.reference.tracker,
        "reference_max_abs_diff": max_diff,
        "reference_ok": ref_ok,
        "known_answer_checks": len(ka),
        "known_answer_passed": int(ka["ok"].sum()),
        "e0_passed": bool(ref_ok and ka["ok"].all()),
    }
    (run / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    pd.set_option("display.width", 160)
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
    print("A) Reference check: MPNTrack on MOT20-train, our wrapper vs original TrackEval")
    print(ref["ours"][cols].round(3).to_string())
    print(
        f"   max |ours - upstream| over all metrics and sequences: {max_diff:.2e}  ->",
        "PASS" if ref_ok else "FAIL",
    )
    print("\nB) Known-answer checks (MOT20-01 shown; all sequences in known_answers.csv)")
    print(ka[ka["seq"] == seqs[0]].drop(columns="seq").to_string(index=False))
    print(
        f"   {summary['known_answer_passed']}/{summary['known_answer_checks']} checks passed "
        f"over {len(seqs)} sequences"
    )
    print(
        f"\nE0 {'PASSED' if summary['e0_passed'] else 'FAILED'} -> {run.relative_to(PROJECT_ROOT)}/"
    )
    if not summary["e0_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
