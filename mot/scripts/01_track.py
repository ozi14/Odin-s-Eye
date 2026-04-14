"""
01_track.py — Phase 1: ByteTrack + DINOv2 Dense Pedestrian Tracking

Processes one or more MOT20 sequences and outputs:
  1. MOTChallenge-format txt  →  output/<seq>/mot_results.txt
  2. Per-frame JSON           →  output/<seq>/frames/frame_XXXXXX.json
  3. Optional visualisation   →  output/<seq>/vis/frame_XXXXXX.jpg

Pipeline per frame:
  YOLO detect → ByteTrack associate (pass1 high-conf + pass2 low-conf)
              → DINOv2 ReID secondary signal
              → write MOT row + JSON

Usage:
    cd mot
    python scripts/01_track.py --seq MOT20-01
    python scripts/01_track.py --seq MOT20-01 MOT20-02 --visualize
    python scripts/01_track.py --seq MOT20-01 --max_frames 100 --no_reid
    python scripts/01_track.py --seq MOT20-01 --device cuda
"""

import os
import sys
import json
import time
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent   # mot/
PROJECT_ROOT = BASE_DIR.parent                      # CV_term_project/
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from odin_eye_mot.tracker.bytetrack import ByteTracker, reset_id_counter
from odin_eye_mot.tracker.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Sequence reader
# ─────────────────────────────────────────────────────────────────────

class MOT20Sequence:
    def __init__(self, seq_dir: Path):
        self.seq_dir = seq_dir
        self.name = seq_dir.name
        self.img_dir = seq_dir / 'img1'
        self.gt_path = seq_dir / 'gt' / 'gt.txt'

        if not self.img_dir.exists():
            raise FileNotFoundError(f"img1/ not found in {seq_dir}")

        self.frames = sorted(self.img_dir.glob('*.jpg'))
        if not self.frames:
            self.frames = sorted(self.img_dir.glob('*.png'))
        if not self.frames:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

        # Read seqinfo.ini for fps + resolution
        self.fps = 30.0
        self.width = self.height = None
        seqinfo = seq_dir / 'seqinfo.ini'
        if seqinfo.exists():
            for line in seqinfo.read_text().splitlines():
                if line.startswith('frameRate'):
                    self.fps = float(line.split('=')[1])
                elif line.startswith('imWidth'):
                    self.width = int(line.split('=')[1])
                elif line.startswith('imHeight'):
                    self.height = int(line.split('=')[1])

        logger.info("Sequence %s: %d frames @ %.0ffps", self.name, len(self.frames), self.fps)

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for frame_path in self.frames:
            frame_id = int(frame_path.stem)
            img = cv2.imread(str(frame_path))
            if img is None:
                logger.warning("Failed to read frame %s", frame_path)
                continue
            yield frame_id, img


# ─────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────

def detect(yolo_model, frame_bgr, conf_thresh=0.3, imgsz=1280):
    """
    Run YOLO on one frame, return list of (bbox_xyxy, score) for class 0 (person).
    """
    results = yolo_model.predict(
        frame_bgr, conf=conf_thresh, classes=[0],
        verbose=False, imgsz=imgsz)

    dets = []
    if results[0].boxes is not None:
        for i in range(len(results[0].boxes)):
            xyxy = results[0].boxes.xyxy[i].cpu().numpy().astype(np.float32)
            conf = float(results[0].boxes.conf[i].cpu())
            dets.append((xyxy, conf))
    return dets


# ─────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────

_PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112, 31),  (255, 178, 29),
    (207, 210,  49), (72,  249, 10),   (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,  212, 187),   (44,  153, 168), (0,  194, 255),
    (52,  69,  147), (100, 115, 255),  (0,   24, 236),  (132,  56, 255),
    (82,   0, 133),  (203, 56,  255),  (255, 149, 200), (255, 55,  199),
]


def _track_color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]


def visualise(frame_bgr, tracks, frame_id: int) -> np.ndarray:
    img = frame_bgr.copy()
    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        color = _track_color(t.track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"T{t.track_id:04d} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f"Frame {frame_id:06d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


# ─────────────────────────────────────────────────────────────────────
# Main tracking loop
# ─────────────────────────────────────────────────────────────────────

def run_sequence(
    seq: MOT20Sequence,
    yolo_model,
    tracker: ByteTracker,
    output_dir: Path,
    max_frames: int   = None,
    conf_thresh: float = 0.3,
    imgsz: int        = 1280,
    visualize: bool   = False,
):
    seq_out   = output_dir / seq.name
    frames_dir = seq_out / 'frames'
    mot_txt   = seq_out / 'mot_results.txt'
    vis_dir   = seq_out / 'vis' if visualize else None

    seq_out.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)
    if visualize:
        vis_dir.mkdir(exist_ok=True)

    mot_rows = []
    total_det = 0
    t0 = time.time()

    frames_iter = list(seq)
    if max_frames:
        frames_iter = frames_iter[:max_frames]

    logger.info("%s", "=" * 60)
    logger.info("Tracking: %s (%d frames)", seq.name, len(frames_iter))
    logger.info("Output: %s", seq_out)
    logger.info("%s", "=" * 60)

    for frame_id, frame_bgr in tqdm(frames_iter, desc=seq.name, unit='fr'):
        # Detection
        dets = detect(yolo_model, frame_bgr, conf_thresh, imgsz)
        total_det += len(dets)

        # ByteTrack update
        active_tracks = tracker.update(dets, frame_bgr)

        # Save MOTChallenge rows
        for t in active_tracks:
            mot_rows.append(t.to_mot_row(frame_id))

        # Save per-frame JSON
        frame_data = {
            "frame_id":    frame_id,
            "n_dets":      len(dets),
            "n_tracks":    len(active_tracks),
            "tracks": [
                {
                    "track_id":   t.track_id,
                    "bbox_xyxy":  [round(float(v), 2) for v in t.bbox_xyxy],
                    "score":      round(t.score, 4),
                    "hits":       t.hits,
                    "age":        t.frame_id - t.start_frame + 1,
                }
                for t in active_tracks
            ]
        }
        with open(frames_dir / f"frame_{frame_id:06d}.json", 'w') as f:
            json.dump(frame_data, f)

        # Visualise
        if visualize:
            vis = visualise(frame_bgr, active_tracks, frame_id)
            cv2.imwrite(str(vis_dir / f"frame_{frame_id:06d}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Write MOTChallenge txt
    mot_rows.sort(key=lambda r: (r[0], r[1]))
    with open(mot_txt, 'w') as f:
        for row in mot_rows:
            f.write(','.join(map(str, row)) + '\n')

    elapsed = time.time() - t0
    fps_eff = len(frames_iter) / max(elapsed, 1e-3)
    n_tracks = len({r[1] for r in mot_rows})

    logger.info("%s complete:", seq.name)
    logger.info("  Frames: %d", len(frames_iter))
    logger.info("  Total dets: %d", total_det)
    logger.info("  Unique IDs: %d", n_tracks)
    logger.info("  MOT txt: %s", mot_txt)
    logger.info("  Speed: %.1f fps (%.1fs total)", fps_eff, elapsed)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1: ByteTrack + DINOv2 dense pedestrian tracking')
    p.add_argument('--seq', nargs='+', required=True,
                   help='Sequence name(s), e.g. MOT20-01 MOT20-02')
    p.add_argument('--dataset_dir', default=str(BASE_DIR / 'datasets' / 'MOT20'),
                   help='Path to MOT20 dataset root')
    p.add_argument('--split', default='train', choices=['train', 'test'],
                   help='Dataset split (default: train)')
    p.add_argument('--output_dir', default=str(BASE_DIR / 'output'),
                   help='Output directory (default: mot/output)')
    p.add_argument('--yolo_weights',
                   default=str(PROJECT_ROOT / 'models' / 'yolo26_weights_v1'
                               / 'best.pt'),
                   help='YOLO weights path (falls back to yolo11x.pt if not found)')
    p.add_argument('--conf_thresh', type=float, default=0.3,
                   help='YOLO confidence threshold (default: 0.3)')
    p.add_argument('--imgsz', type=int, default=1280,
                   help='YOLO inference image size (default: 1280)')
    p.add_argument('--thresh_high', type=float, default=0.6,
                   help='ByteTrack high-conf threshold (default: 0.6)')
    p.add_argument('--thresh_low', type=float, default=0.1,
                   help='ByteTrack low-conf threshold (default: 0.1)')
    p.add_argument('--max_lost_frames', type=int, default=30,
                   help='Frames before a lost track is removed (default: 30)')
    p.add_argument('--no_reid', action='store_true',
                   help='Disable DINOv2 ReID (IoU-only association)')
    p.add_argument('--reid_model', default='dinov2_vitb14_reg',
                   choices=['dinov2_vitb14_reg', 'dinov2_vitl14_reg',
                            'dinov2_vits14_reg'],
                   help='DINOv2 model variant (default: dinov2_vitb14_reg)')
    p.add_argument('--reid_weight', type=float, default=0.3,
                   help='ReID vs IoU blend weight: 0=IoU only, 1=ReID only (default: 0.3)')
    p.add_argument('--reid_thresh', type=float, default=0.3,
                   help='Minimum cosine similarity for ReID match (default: 0.3)')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Process at most N frames per sequence (for testing)')
    p.add_argument('--visualize', action='store_true',
                   help='Save annotated frame images')
    p.add_argument('--device', default=None,
                   help='Device: cuda / mps / cpu (auto-detect if omitted)')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── device ──────────────────────────────────────────────────────
    import torch
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info("Device: %s", device)

    # ── YOLO ────────────────────────────────────────────────────────
    yolo_path = args.yolo_weights
    if not os.path.exists(yolo_path):
        logger.warning("Custom weights not found at %s", yolo_path)
        yolo_path = 'yolo11x.pt'
        logger.info("Falling back to pretrained weights: %s", yolo_path)
    yolo = YOLO(yolo_path)

    # ── ReID (optional) ─────────────────────────────────────────────
    reid_extractor = None
    if not args.no_reid:
        try:
            from odin_eye_mot.reid.dinov2_extractor import DINOv2ReIDExtractor
            reid_extractor = DINOv2ReIDExtractor(
                model_name=args.reid_model, device=device)
        except (ImportError, RuntimeError, OSError) as err:
            logger.warning("DINOv2 ReID unavailable (%s). Running IoU-only.", err)

    # ── process sequences ────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in args.seq:
        seq_path = dataset_dir / args.split / seq_name
        if not seq_path.exists():
            logger.warning("Sequence not found: %s", seq_path)
            continue

        seq = MOT20Sequence(seq_path)

        # Fresh tracker per sequence
        reset_id_counter()
        tracker = ByteTracker(
            thresh_high=args.thresh_high,
            thresh_low=args.thresh_low,
            max_lost_frames=args.max_lost_frames,
            reid_extractor=reid_extractor,
            reid_weight=args.reid_weight,
            reid_thresh=args.reid_thresh,
        )

        run_sequence(
            seq=seq,
            yolo_model=yolo,
            tracker=tracker,
            output_dir=output_dir,
            max_frames=args.max_frames,
            conf_thresh=args.conf_thresh,
            imgsz=args.imgsz,
            visualize=args.visualize,
        )

    logger.info("All sequences done. Results in %s/", output_dir)


if __name__ == '__main__':
    main()
