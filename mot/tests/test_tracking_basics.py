from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from odin_eye_mot.tracker.bytetrack import iou_matrix, linear_assignment


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "03_benchmark.py"
    spec = spec_from_file_location("benchmark_script", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_iou_matrix_identity_box():
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = iou_matrix(box, box)
    assert iou.shape == (1, 1)
    assert iou[0, 0] == 1.0


def test_linear_assignment_threshold_filters_match():
    cost = np.array([[0.2, 0.8], [0.9, 0.3]], dtype=np.float32)
    matched, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=0.4)
    assert matched.shape == (2, 2)
    assert unmatched_rows.size == 0
    assert unmatched_cols.size == 0


def test_load_mot_txt_skips_malformed_and_negative_conf(tmp_path):
    mod = _load_benchmark_module()
    mot_txt = tmp_path / "mot_results.txt"
    mot_txt.write_text(
        "\n".join(
            [
                "1,10,10,20,30,40,0.9,-1,-1,-1",
                "2,11,20,30,40,50,-1,-1,-1,-1",
                "bad,row,should,be,ignored",
                "3,12,5,5,10,10,0.5,-1,-1,-1",
            ]
        )
    )
    data = mod.load_mot_txt(mot_txt)
    assert sorted(data.keys()) == [1, 3]
    assert len(data[1]) == 1
    assert data[1][0][0] == 10

