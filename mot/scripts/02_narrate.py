"""
02_narrate.py — Phase 2: VLM Scene Narration (Qwen3-VL)

Reads tracking results from Phase 1 and invokes Qwen3-VL to generate
structured natural-language descriptions every N frames.

Input:
  mot/output/<seq>/frames/frame_XXXXXX.json  — per-frame track data
  mot/datasets/MOT20/train/<seq>/img1/       — original frames

Output:
  mot/output/<seq>/narration/
    ├── narration_XXXXXX.json    — per-narration JSON
    └── narration_summary.json  — all narrations in one file

Narration modes (can be mixed per run):
  - scene_summary   (default, every --narrate_every frames)
  - person_describe (on demand: --highlight_id T005)
  - interaction
  - anomaly

Usage:
    cd mot
    python scripts/02_narrate.py --seq MOT20-01 --backend mlx
    python scripts/02_narrate.py --seq MOT20-01 --backend transformers --device cuda
    python scripts/02_narrate.py --seq MOT20-01 --mode anomaly --narrate_every 60
    python scripts/02_narrate.py --seq MOT20-01 --max_frames 300 --backend mlx
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────
# Lightweight mock Track for JSON → Track reconstruction
# ─────────────────────────────────────────────────────────────────────

class _MockTrack:
    """Reconstruct a minimal Track-like object from JSON data."""
    def __init__(self, track_id: int, bbox_xyxy, score: float):
        import numpy as np
        self.track_id = track_id
        self.bbox_xyxy = np.array(bbox_xyxy, dtype=np.float32) if bbox_xyxy else None
        self.score = score


def load_tracks_from_json(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    tracks = []
    for t in data.get('tracks', []):
        tracks.append(_MockTrack(
            track_id=t['track_id'],
            bbox_xyxy=t.get('bbox_xyxy'),
            score=t.get('score', 1.0),
        ))
    return tracks, data.get('frame_id', 0)


# ─────────────────────────────────────────────────────────────────────
# Main narration loop
# ─────────────────────────────────────────────────────────────────────

def run_narration(
    seq_name:     str,
    dataset_dir:  Path,
    output_dir:   Path,
    split:        str   = 'train',
    backend:      str   = 'mlx',
    model_id:     str   = None,
    device:       str   = 'cuda',
    mode:         str   = 'scene_summary',
    narrate_every: int  = 30,
    max_frames:   int   = None,
    highlight_id: int   = None,
):
    from odin_eye_mot.vlm.narrator import Narrator

    seq_out = output_dir / seq_name
    frames_dir = seq_out / 'frames'
    img_dir   = dataset_dir / split / seq_name / 'img1'
    narr_dir  = seq_out / 'narration'
    narr_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        print(f"ERROR: Tracking results not found at {frames_dir}")
        print("Run 01_track.py first.")
        return

    if not img_dir.exists():
        print(f"ERROR: Image directory not found at {img_dir}")
        return

    # Collect frame JSON files
    frame_jsons = sorted(frames_dir.glob('frame_*.json'))
    if max_frames:
        frame_jsons = frame_jsons[:max_frames]

    # Read fps from seqinfo.ini
    fps = 30.0
    seqinfo = dataset_dir / split / seq_name / 'seqinfo.ini'
    if seqinfo.exists():
        for line in seqinfo.read_text().splitlines():
            if line.startswith('frameRate'):
                fps = float(line.split('=')[1])

    print(f"\n{'='*60}")
    print(f"  Narration: {seq_name}  ({len(frame_jsons)} frames)")
    print(f"  Mode:      {mode}  every {narrate_every} frames")
    print(f"  Backend:   {backend}")
    print(f"{'='*60}")

    narrator = Narrator(
        backend=backend,
        model_id=model_id,
        narrate_every=narrate_every,
        device=device,
    )

    all_results = []
    n_narrated  = 0

    for jpath in tqdm(frame_jsons, desc='Narrating', unit='fr'):
        tracks, frame_id = load_tracks_from_json(jpath)

        if not narrator.should_narrate(frame_id):
            continue

        # Load original frame
        img_path = img_dir / f"{frame_id:06d}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{frame_id:06d}.png"
        if not img_path.exists():
            print(f"WARNING: Image not found for frame {frame_id}")
            continue

        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            continue

        result = narrator.narrate(
            frame_bgr=frame_bgr,
            tracks=tracks,
            mode=mode,
            frame_id=frame_id,
            fps=fps,
            highlight_id=highlight_id,
        )
        all_results.append(result)
        n_narrated += 1

        # Save individual narration
        out_path = narr_dir / f"narration_{frame_id:06d}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

    # Save summary
    summary_path = narr_dir / 'narration_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            "sequence":    seq_name,
            "mode":        mode,
            "backend":     backend,
            "n_narrations": n_narrated,
            "fps":         fps,
            "narrations":  all_results,
        }, f, indent=2)

    print(f"\n  Narrated {n_narrated} frames.")
    print(f"  Summary:  {summary_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 2: VLM Scene Narration with Qwen3-VL')
    p.add_argument('--seq', required=True,
                   help='Sequence name, e.g. MOT20-01')
    p.add_argument('--dataset_dir', default=str(BASE_DIR / 'datasets' / 'MOT20'))
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--output_dir', default=str(BASE_DIR / 'output'))
    p.add_argument('--backend', default='mlx', choices=['mlx', 'transformers'],
                   help='VLM inference backend (default: mlx)')
    p.add_argument('--model_id', default=None,
                   help='Override default model ID')
    p.add_argument('--device', default='cuda',
                   help='Device for transformers backend (default: cuda)')
    p.add_argument('--mode', default='scene_summary',
                   choices=['scene_summary', 'person_describe',
                            'interaction', 'anomaly'],
                   help='Narration mode (default: scene_summary)')
    p.add_argument('--narrate_every', type=int, default=30,
                   help='Narrate every N frames (default: 30)')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Process at most N frames (for testing)')
    p.add_argument('--highlight_id', type=int, default=None,
                   help='Track ID to highlight for person_describe mode')
    return p.parse_args()


def main():
    args = parse_args()
    run_narration(
        seq_name=args.seq,
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        split=args.split,
        backend=args.backend,
        model_id=args.model_id,
        device=args.device,
        mode=args.mode,
        narrate_every=args.narrate_every,
        max_frames=args.max_frames,
        highlight_id=args.highlight_id,
    )


if __name__ == '__main__':
    main()
