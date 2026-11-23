"""E1: score one tracker on all four MOT20 training sequences.

Usage:
    python scripts/90_evaluate.py predictions_dir=data/interim/tracks/mytracker
    python scripts/90_evaluate.py simulate_gt=true

The run directory holds metrics.csv and two density tables, frame_density.csv
and grid_density.csv. All three include sequence and COMBINED results.
The density tables use GT occupancy. HOTA and IDF1 are recomputed on each bin's
observations. IDSW is attributed from full-sequence CLEAR matches. HOTA and IDF1
are blank when both GT and predictions are absent in a bin.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from odins_eye.eval.density import evaluate_density_bins
from odins_eye.eval.sanity import scored_gt
from odins_eye.eval.trackeval_runner import COMBINED, evaluate_tracks
from odins_eye.io.mot import read_mot_gt
from odins_eye.io.schemas import TRACKS_SCHEMA, read_table
from odins_eye.utils.paths import PROJECT_ROOT
from odins_eye.utils.run import new_run_dir


@hydra.main(version_base=None, config_path="../configs", config_name="e1")
def main(cfg: DictConfig) -> None:
    if cfg.simulate_gt == (cfg.predictions_dir is not None):
        raise SystemExit("Set exactly one of predictions_dir=<directory> or simulate_gt=true.")

    gt_folder = PROJECT_ROOT / cfg.data.raw_dir / cfg.data.split
    if cfg.simulate_gt:
        tracker = "gt-copy"
        tracks_by_seq = {
            seq: scored_gt(read_mot_gt(gt_folder / seq, anchor=cfg.data.anchor))
            for seq in cfg.data.sequences
        }
    else:
        predictions_dir = (PROJECT_ROOT / cfg.predictions_dir).resolve()
        tracker = predictions_dir.name
        tracks_by_seq = {}
        for seq in cfg.data.sequences:
            path = predictions_dir / f"{seq}.parquet"
            if not path.is_file():
                raise SystemExit(f"Missing prediction file: {path}")
            tracks = read_table(path, TRACKS_SCHEMA)
            if (tracks["video_id"] != seq).any():
                raise SystemExit(f"{path}: every video_id must match {seq}.")
            tracks_by_seq[seq] = tracks

    run = new_run_dir(cfg.experiment.id, tracker, cfg)
    metrics = evaluate_tracks(tracks_by_seq, gt_folder, run / "work", tracker, cfg.data.name)
    metrics_path = run / "metrics.csv"
    metrics.to_csv(metrics_path)
    print(metrics.round(3).to_string())

    frame_density, grid_density = evaluate_density_bins(
        run / "work",
        tracker,
        gt_folder,
        metrics,
        cfg.data.name,
        frame_bins=list(cfg.density.frame_bins),
        grid_bins=list(cfg.density.grid_bins),
        rows=cfg.density.grid_rows,
        cols=cfg.density.grid_cols,
    )
    for name, table in (("frame_density", frame_density), ("grid_density", grid_density)):
        table.to_csv(run / f"{name}.csv")
        print(f"\n{name} ({COMBINED})\n{table.loc[COMBINED].to_string()}")
    print(f"\nResults saved to {run.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
