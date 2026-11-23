import pandas as pd
import pytest

from odins_eye.eval.density import evaluate_density_bins
from odins_eye.eval.trackeval_runner import COMBINED, evaluate_tracks
from odins_eye.io.schemas import TRACKS_SCHEMA


def test_density_counts_match_across_grid_edges_and_include_empty_cells(toy_seq, tmp_path):
    (toy_seq / "gt" / "gt.txt").write_text(
        "1,1,12,3,6,4,1,1,1\n1,2,2,15,4,4,0,7,1\n2,1,20,5,4,4,1,1,1\n"
    )
    seq = toy_seq.name
    predictions = pd.DataFrame(
        [
            (seq, 1, 1, 14, 3, 20, 7, 1, "body"),  # IoU .5, in the adjacent cell
            (seq, 1, 2, 2, 15, 6, 19, 1, "body"),  # distractor, ignored
            (seq, 2, 3, 1, 16, 4, 20, 1, "body"),  # FP in an empty cell
            (seq, 3, 4, 20, 16, 24, 20, 1, "body"),  # FP in an empty frame
        ],
        columns=TRACKS_SCHEMA.names,
    )
    work = tmp_path / "mot"
    metrics = evaluate_tracks({seq: predictions}, toy_seq.parent, work)
    frames, grid = evaluate_density_bins(
        work,
        "tracker",
        toy_seq.parent,
        metrics,
        "MOT20",
        frame_bins=[0, 1],
        grid_bins=[0, 1],
        rows=2,
        cols=2,
    )

    columns = ["samples", "gt_people", "FN", "FP", "IDSW"]
    assert frames.loc[COMBINED, columns].to_dict("index") == {
        "0": {"samples": 1, "gt_people": 0, "FN": 0, "FP": 1, "IDSW": 0},
        "1+": {"samples": 2, "gt_people": 2, "FN": 1, "FP": 1, "IDSW": 0},
    }
    assert grid.loc[COMBINED, columns].to_dict("index") == {
        "0": {"samples": 10, "gt_people": 0, "FN": 0, "FP": 2, "IDSW": 0},
        "1+": {"samples": 2, "gt_people": 2, "FN": 1, "FP": 0, "IDSW": 0},
    }
    assert grid.loc[(COMBINED, "0"), ["HOTA", "IDF1"]].tolist() == [0, 0]
    assert grid.loc[(COMBINED, "1+"), "IDF1"] == pytest.approx(200 / 3)


def test_switch_after_gap_counts_in_destination_frame_and_grid_bin(toy_seq, tmp_path):
    (toy_seq / "gt" / "gt.txt").write_text(
        "1,1,2,3,4,4,1,1,1\n3,1,18,3,4,4,1,1,1\n3,2,26,3,4,4,1,1,1\n"
    )
    seq = toy_seq.name
    predictions = pd.DataFrame(
        [
            (seq, 1, 10, 2, 3, 6, 7, 1, "body"),
            (seq, 3, 11, 18, 3, 22, 7, 1, "body"),
            (seq, 3, 20, 26, 3, 30, 7, 1, "body"),
        ],
        columns=TRACKS_SCHEMA.names,
    )
    work = tmp_path / "mot"
    metrics = evaluate_tracks({seq: predictions}, toy_seq.parent, work)
    frames, grid = evaluate_density_bins(
        work,
        "tracker",
        toy_seq.parent,
        metrics,
        "MOT20",
        frame_bins=[0, 1, 2],
        grid_bins=[0, 1, 2],
        rows=1,
        cols=2,
    )
    for table in (frames, grid):
        combined = table.loc[COMBINED]
        assert combined["IDSW"].tolist() == [0, 0, 1]
        assert combined.loc["0", ["HOTA", "IDF1"]].isna().all()
        assert (combined.loc[["1", "2+"], ["HOTA", "IDF1"]] == 100).all().all()
    assert metrics.loc[COMBINED, "IDF1"] < 100

    # With one bin the identity history still spans both cells and the empty frame.
    frames, grid = evaluate_density_bins(
        work,
        "tracker",
        toy_seq.parent,
        metrics,
        "MOT20",
        frame_bins=[0],
        grid_bins=[0],
        rows=1,
        cols=2,
    )
    columns = ["HOTA", "IDF1", "IDSW"]
    for table in (frames, grid):
        assert table.loc[(COMBINED, "0+"), columns].tolist() == pytest.approx(
            metrics.loc[COMBINED, columns].tolist()
        )
