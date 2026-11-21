"""Runs the real detector on 2 MOT20 frames (skipped without weights or data)."""

import pytest

from odins_eye.perception.detector import DetectorConfig, YoloDetector, detect_sequence
from odins_eye.utils.paths import DATA_DIR

WEIGHTS = DATA_DIR / "weights" / "yolo26m_ft1_v3_1280.pt"
SEQ = DATA_DIR / "raw" / "MOT20" / "train" / "MOT20-01"
pytestmark = pytest.mark.skipif(
    not (WEIGHTS.exists() and SEQ.exists()), reason="weights or MOT20 missing"
)


def test_detects_people_on_first_frames():
    cfg = DetectorConfig(name="t", weights=str(WEIGHTS), batch=2)
    dets, stats = detect_sequence(YoloDetector(cfg), SEQ, max_frames=2)
    assert stats["frames"] == 2 and set(dets["frame"]) == {1, 2}
    assert 30 <= stats["dets_per_frame_mean"] <= 70  # ~46 GT pedestrians per frame
    assert dets["score"].min() >= 0.1 and stats["frames_hitting_max_det"] == 0
