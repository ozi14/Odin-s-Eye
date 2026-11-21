"""Read MOTChallenge `seqinfo.ini`: the single source of truth for fps, length and size."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeqInfo:
    name: str
    seq_dir: Path
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_dir: str = "img1"
    im_ext: str = ".jpg"

    @property
    def image_dir(self) -> Path:
        return self.seq_dir / self.im_dir


def read_seqinfo(seq_dir: str | Path) -> SeqInfo:
    seq_dir = Path(seq_dir)
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        raise FileNotFoundError(f"no seqinfo.ini in {seq_dir}")
    parser = configparser.ConfigParser()
    parser.read(ini)
    s = parser["Sequence"]  # configparser keys are case-insensitive
    return SeqInfo(
        name=s["name"],
        seq_dir=seq_dir,
        frame_rate=float(s["frameRate"]),
        seq_length=int(s["seqLength"]),
        im_width=int(s["imWidth"]),
        im_height=int(s["imHeight"]),
        im_dir=s.get("imDir", "img1"),
        im_ext=s.get("imExt", ".jpg"),
    )
