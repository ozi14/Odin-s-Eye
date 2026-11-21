"""Checks against the real MOT20 files (skipped if data/raw/MOT20 is not present)."""

import pytest

from odins_eye.io.frames import iter_frames
from odins_eye.io.mot import read_mot_gt
from odins_eye.utils.paths import DATA_DIR

SEQ = DATA_DIR / "raw" / "MOT20" / "train" / "MOT20-01"
pytestmark = pytest.mark.skipif(not SEQ.exists(), reason="MOT20 not downloaded")


def test_mot20_01_ground_truth_matches_known_counts():
    gt = read_mot_gt(SEQ)
    assert len(gt) == 26647
    assert gt["frame"].min() == 1 and gt["frame"].max() == 429
    ped = gt[gt["consider"] & (gt["mot_class"] == 1)]
    per_frame = ped.groupby("frame").size()
    assert 45 <= per_frame.mean() <= 47  # ~46 pedestrians per frame


def test_mot20_01_first_frame_streams_at_full_hd():
    idx, img = next(iter_frames(SEQ))
    assert idx == 1 and img.shape == (1080, 1920, 3)
