"""M1 detector: YOLO26m person detector -> detections-1.0 (ADR-003).

GPU-independent by design: the same code runs on H200, A100, MPS or CPU; only `device` and
`half` change, and both are resolved automatically. Results can differ in the last float digits
between GPU types, so every run records the GPU it ran on, and all trackers read the SAME saved
detections.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odins_eye.io.frames import iter_frames
from odins_eye.io.schemas import DETECTIONS_SCHEMA, to_table
from odins_eye.io.seqinfo import read_seqinfo


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    weights: str
    imgsz: int = 1280  # same size as training (ft1_v3 was trained at 1280)
    conf: float = 0.1  # keep weak boxes for ByteTrack's second pass
    iou: float = 0.7  # NMS IoU: this checkpoint's head is not end-to-end, so NMS runs
    max_det: int = 1000  # Ultralytics default 300 could truncate dense frames
    batch: int = 16
    half: bool = True  # fp16, CUDA only (ignored on MPS/CPU)
    device: str = "auto"  # auto -> cuda:0 | mps | cpu
    classes: tuple[int, ...] = (0,)
    anchor: str = "body"


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apostrophe_safe_path(path: str | Path) -> str:
    """Return a path Ultralytics can open.

    `ultralytics.utils.downloads.attempt_download_asset` deletes every "'" from the weights path
    ("Odin's Eye v4" -> "Odins Eye v4"), so such paths are loaded through a symlink in the temp dir.
    """
    path = str(Path(path).resolve())
    if "'" not in path:
        return path
    tag = hashlib.sha1(path.encode()).hexdigest()[:10]
    link = Path(tempfile.gettempdir()) / f"odins_eye_{tag}{Path(path).suffix}"
    if "'" in str(link):
        raise RuntimeError(f"temp dir path contains an apostrophe: {link}")
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(path)
    return str(link)


def boxes_to_frame(
    video_id: str, frame: int, boxes: np.ndarray, anchor: str = "body"
) -> pd.DataFrame:
    """[N, 5] array (x1, y1, x2, y2, score) -> detections-1.0 rows, det_id by descending score.

    Degenerate boxes (zero width or height after clipping) are dropped.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    boxes = boxes[np.argsort(-boxes[:, 4], kind="stable")]
    return pd.DataFrame(
        {
            "video_id": video_id,
            "frame": np.full(len(boxes), frame, dtype=np.int32),
            "det_id": np.arange(len(boxes), dtype=np.int32),
            "x1": boxes[:, 0],
            "y1": boxes[:, 1],
            "x2": boxes[:, 2],
            "y2": boxes[:, 3],
            "score": boxes[:, 4],
            "anchor": anchor,
        }
    )


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.half = bool(cfg.half and self.device.startswith("cuda"))
        self.model = YOLO(apostrophe_safe_path(cfg.weights))
        self.end2end = bool(getattr(self.model.model.model[-1], "end2end", False))

    def settings(self) -> dict:
        return {
            **asdict(self.cfg),
            "device": self.device,
            "half": self.half,
            "end2end": self.end2end,
        }

    def detect(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Run on a batch of BGR frames; returns one [N, 5] array per frame."""
        results = self.model.predict(
            frames,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            classes=list(self.cfg.classes),
            half=self.half,
            device=self.device,
            batch=len(frames),
            verbose=False,
        )
        out = []
        for r in results:
            xyxy = r.boxes.xyxy.float().cpu().numpy()
            score = r.boxes.conf.float().cpu().numpy()[:, None]
            out.append(np.concatenate([xyxy, score], axis=1))
        return out


def _batches(it: Iterator, size: int) -> Iterator[list]:
    batch = []
    for item in it:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def detect_sequence(
    detector: YoloDetector, seq_dir: str | Path, max_frames: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Detect people in every frame of one sequence (streamed); return detections and timing."""
    info = read_seqinfo(seq_dir)
    end = info.seq_length if max_frames is None else min(max_frames, info.seq_length)
    parts, model_s, n_frames, capped = [], 0.0, 0, 0
    t0 = time.perf_counter()
    for batch in _batches(iter_frames(seq_dir, 1, end), detector.cfg.batch):
        idx, imgs = zip(*batch, strict=True)
        t = time.perf_counter()
        boxes = detector.detect(list(imgs))
        model_s += time.perf_counter() - t
        for frame, b in zip(idx, boxes, strict=True):
            capped += int(len(b) >= detector.cfg.max_det)
            parts.append(boxes_to_frame(info.name, frame, b, detector.cfg.anchor))
        n_frames += len(idx)
    total_s = time.perf_counter() - t0
    dets = pd.concat(parts, ignore_index=True)
    dets = to_table(dets, DETECTIONS_SCHEMA).to_pandas()  # validate before anyone reads it
    per_frame = dets.groupby("frame").size().reindex(range(1, end + 1), fill_value=0)
    stats = {
        "seq": info.name,
        "frames": n_frames,
        "detections": len(dets),
        "dets_per_frame_mean": round(float(per_frame.mean()), 2),
        "dets_per_frame_max": int(per_frame.max()),
        "frames_hitting_max_det": capped,
        "seconds_total": round(total_s, 2),
        "fps_total": round(n_frames / total_s, 2),
        "fps_model": round(n_frames / model_s, 2),
    }
    return dets, stats
