import numpy as np
import pytest

from odins_eye.io.frames import iter_frames
from odins_eye.io.mot import read_mot_gt, read_mot_results, write_mot_results
from odins_eye.io.seqinfo import read_seqinfo


def test_seqinfo_fields(toy_seq):
    info = read_seqinfo(toy_seq)
    assert (info.name, info.frame_rate, info.seq_length) == ("TOY-01", 25.0, 3)
    assert (info.im_width, info.im_height) == (32, 24)


def test_gt_is_read_unfiltered_with_corner_boxes(toy_seq):
    gt = read_mot_gt(toy_seq)
    assert len(gt) == 3  # the static person (class 7, consider 0) is kept
    first = gt.iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (199, 813, 339, 1081)
    assert list(gt["mot_class"]) == [1, 7, 1]
    assert list(gt["consider"]) == [True, False, True]


def test_mot_results_round_trip(toy_seq, tmp_path):
    gt = read_mot_gt(toy_seq)
    path = write_mot_results(gt, tmp_path / "out" / "TOY-01.txt")
    assert path.read_text().splitlines()[0] == "1,1,199.00,813.00,140.00,268.00,1.0000,-1,-1,-1"
    back = read_mot_results(path, video_id="TOY-01")
    cols = ["frame", "track_id", "x1", "y1", "x2", "y2", "score"]
    assert np.allclose(back[cols].to_numpy(float), gt[cols].to_numpy(float))


def test_frames_stream_in_order_with_seqinfo_size(toy_seq):
    got = [(i, img.shape) for i, img in iter_frames(toy_seq)]
    assert got == [(1, (24, 32, 3)), (2, (24, 32, 3)), (3, (24, 32, 3))]
    assert [i for i, _ in iter_frames(toy_seq, start=2)] == [2, 3]


def test_missing_frame_fails_loudly(toy_seq):
    (toy_seq / "img1" / "000002.jpg").unlink()
    with pytest.raises(ValueError, match="seqinfo says 3"):
        next(iter_frames(toy_seq))
