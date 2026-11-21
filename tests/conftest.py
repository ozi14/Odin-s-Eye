from pathlib import Path

import cv2
import numpy as np
import pytest


def make_toy_seq(root: Path, n_frames: int = 3, w: int = 32, h: int = 24) -> Path:
    """A tiny MOTChallenge sequence: 3 frames, 2 pedestrian rows and 1 static person."""
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


@pytest.fixture
def toy_seq(tmp_path) -> Path:
    return make_toy_seq(tmp_path)
