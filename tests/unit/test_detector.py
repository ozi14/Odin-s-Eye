import numpy as np
import pytest

from odins_eye.io.schemas import DETECTIONS_SCHEMA, SchemaError, to_table
from odins_eye.perception.detector import boxes_to_frame


def test_boxes_become_detections_sorted_by_score():
    boxes = np.array([[0, 0, 10, 20, 0.2], [5, 5, 15, 30, 0.9], [1, 1, 1, 9, 0.5]])  # last: w=0
    df = boxes_to_frame("V", 7, boxes)
    assert np.allclose(df["score"], [0.9, 0.2])  # degenerate box dropped, sorted by score
    assert list(df["det_id"]) == [0, 1] and set(df["frame"]) == {7}
    to_table(df, DETECTIONS_SCHEMA)  # valid detections-1.0


def test_empty_frame_is_valid():
    df = boxes_to_frame("V", 1, np.zeros((0, 5)))
    assert len(to_table(df, DETECTIONS_SCHEMA)) == 0


def test_duplicate_det_id_is_rejected():
    df = boxes_to_frame("V", 1, np.array([[0, 0, 10, 20, 0.2], [5, 5, 15, 30, 0.9]]))
    df["det_id"] = 0
    with pytest.raises(SchemaError, match="det_id"):
        to_table(df, DETECTIONS_SCHEMA)


def test_apostrophe_paths_are_made_loadable(tmp_path):
    from odins_eye.perception.detector import apostrophe_safe_path

    weights = tmp_path / "Odin's Eye" / "w.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"x")
    safe = apostrophe_safe_path(weights)
    assert "'" not in safe and open(safe, "rb").read() == b"x"
