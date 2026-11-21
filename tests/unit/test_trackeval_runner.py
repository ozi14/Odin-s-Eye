"""Runs on the toy sequence, so it needs no downloaded data (works in CI)."""

from odins_eye.eval.sanity import scored_gt
from odins_eye.eval.trackeval_runner import COMBINED, evaluate_tracks
from odins_eye.io.mot import read_mot_gt


def test_gt_copy_scores_perfect_on_toy_sequence(toy_seq, tmp_path):
    gt = read_mot_gt(toy_seq)
    res = evaluate_tracks({"TOY-01": scored_gt(gt)}, toy_seq.parent, tmp_path, "gt_copy")
    assert list(res.index) == ["TOY-01", COMBINED]
    row = res.loc["TOY-01"]
    assert row["HOTA"] == 100 and row["MOTA"] == 100 and row["IDF1"] == 100
    assert row["GT_Dets"] == 2  # the static person is not a scored pedestrian


def test_prediction_on_static_person_is_not_a_false_positive(toy_seq, tmp_path):
    gt = read_mot_gt(toy_seq)
    preds = gt.copy()  # includes an exact box on the static person (class 7)
    res = evaluate_tracks({"TOY-01": preds}, toy_seq.parent, tmp_path, "with_static")
    assert res.loc["TOY-01", "FP"] == 0 and res.loc["TOY-01", "HOTA"] == 100
