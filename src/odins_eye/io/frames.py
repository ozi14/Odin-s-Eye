"""Stream frames one at a time. v3 loaded whole sequences into RAM (~17 GB for MOT20-02)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from odins_eye.io.seqinfo import read_seqinfo


def iter_frames(
    seq_dir: str | Path, start: int = 1, end: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR uint8 image [H, W, 3]) for frames start..end (1-based, inclusive).

    Checks that the image folder matches seqinfo.ini (count, contiguous names, image size),
    so a missing or extra frame fails loudly instead of shifting every later frame.
    """
    info = read_seqinfo(seq_dir)
    files = sorted(info.image_dir.glob(f"*{info.im_ext}"))
    if len(files) != info.seq_length:
        raise ValueError(
            f"{info.name}: {len(files)} images in {info.image_dir}, seqinfo says {info.seq_length}"
        )
    end = info.seq_length if end is None else min(end, info.seq_length)
    if not 1 <= start <= end:
        raise ValueError(f"invalid frame range {start}..{end}")
    for idx in range(start, end + 1):
        path = files[idx - 1]
        if int(path.stem) != idx:
            raise ValueError(f"{info.name}: expected frame {idx}, found {path.name}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read {path}")
        if img.shape[:2] != (info.im_height, info.im_width):
            raise ValueError(f"{path.name}: size {img.shape[1]}x{img.shape[0]} != seqinfo")
        yield idx, img
