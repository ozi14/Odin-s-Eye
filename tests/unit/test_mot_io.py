from pathlib import Path

import cv2
import numpy as np
import pytest

from odins_eye.io.frames import iter_frames
from odins_eye.io.mot import read_mot_gt, read_mot_results, write_mot_results
from odins_eye.io.seqinfo import read_seqinfo


def _make_seq(root: Path, n_frames: int = 3, w: int = 32, h: int = 24) -> Path:
    seq = root / "TOY-01"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=TOY-01\nimDir=img1\nframeRate=25\n"
        f"seqLength={n_frames}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
    )
    for i in range(1, n_frames + 1):
        cv2.imwrite(str(seq / "img1" / f"{i:06d}.jpg"), np.full((h, w, 3), i * 10, np.uint8))
    # frame,id,left,top,width,height,consider,class,visibility
    (seq / "gt" / "gt.txt").write_text(
        "1,1,199,813,140,268,1,1,0.83\n1,2,1013,201,46,104,0,7,0.42\n2,1,201,812,139,269,1,1,0.85\n"
    )
    return seq


def test_seqinfo_fields(tmp_path):
    info = read_seqinfo(_make_seq(tmp_path))
    assert (info.name, info.frame_rate, info.seq_length) == ("TOY-01", 25.0, 3)
    assert (info.im_width, info.im_height) == (32, 24)


def test_gt_is_read_unfiltered_with_corner_boxes(tmp_path):
    gt = read_mot_gt(_make_seq(tmp_path))
    assert len(gt) == 3  # the static person (class 7, consider 0) is kept
    first = gt.iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (199, 813, 339, 1081)
    assert list(gt["mot_class"]) == [1, 7, 1]
    assert list(gt["consider"]) == [True, False, True]


def test_mot_results_round_trip(tmp_path):
    gt = read_mot_gt(_make_seq(tmp_path))
    path = write_mot_results(gt, tmp_path / "out" / "TOY-01.txt")
    assert path.read_text().splitlines()[0] == "1,1,199.00,813.00,140.00,268.00,1.0000,-1,-1,-1"
    back = read_mot_results(path, video_id="TOY-01")
    cols = ["frame", "track_id", "x1", "y1", "x2", "y2", "score"]
    assert np.allclose(back[cols].to_numpy(float), gt[cols].to_numpy(float))


def test_frames_stream_in_order_with_seqinfo_size(tmp_path):
    seq = _make_seq(tmp_path)
    got = [(i, img.shape) for i, img in iter_frames(seq)]
    assert got == [(1, (24, 32, 3)), (2, (24, 32, 3)), (3, (24, 32, 3))]
    assert [i for i, _ in iter_frames(seq, start=2)] == [2, 3]


def test_missing_frame_fails_loudly(tmp_path):
    seq = _make_seq(tmp_path)
    (seq / "img1" / "000002.jpg").unlink()
    with pytest.raises(ValueError, match="seqinfo says 3"):
        next(iter_frames(seq))
