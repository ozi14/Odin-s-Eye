"""Known-answer checks on real MOT20-01 (skipped if the data is not present)."""

import pytest

from odins_eye.eval.sanity import known_answer_suite
from odins_eye.io.mot import read_mot_gt
from odins_eye.utils.paths import DATA_DIR

GT_FOLDER = DATA_DIR / "raw" / "MOT20" / "train"
pytestmark = pytest.mark.skipif(not GT_FOLDER.exists(), reason="MOT20 not downloaded")


def test_all_known_answers_hold_on_mot20_01(tmp_path):
    ka = known_answer_suite(read_mot_gt(GT_FOLDER / "MOT20-01"), GT_FOLDER, tmp_path)
    failed = ka[~ka["ok"]]
    assert failed.empty, failed.to_string()
