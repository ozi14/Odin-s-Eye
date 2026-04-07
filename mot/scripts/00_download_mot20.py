"""
00_download_mot20.py — Download and set up MOT20 dataset

Downloads MOT20 (4 train + 4 test sequences) from Kaggle and structures
the directory for use by the tracking pipeline.

Expected output layout:
    mot/datasets/MOT20/
    ├── train/
    │   ├── MOT20-01/  (429 frames, night indoor, ~75 ped/frame)
    │   ├── MOT20-02/  (2782 frames, day outdoor, ~150 ped/frame)
    │   ├── MOT20-03/  (2405 frames, night outdoor, ~125 ped/frame)
    │   └── MOT20-05/  (3315 frames, day outdoor, ~150 ped/frame)
    └── test/
        ├── MOT20-04/
        ├── MOT20-06/
        ├── MOT20-07/
        └── MOT20-08/

Each sequence follows MOTChallenge format:
    MOT20-XX/
    ├── seqinfo.ini
    ├── img1/          (frames: 000001.jpg …)
    └── gt/
        └── gt.txt     (ground-truth, train only)

Usage:
    cd mot
    python scripts/00_download_mot20.py
    python scripts/00_download_mot20.py --dest datasets/MOT20
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # mot/


def download_kaggle(dest: Path):
    """Download MOT20 via kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed.  Run: pip install kagglehub")
        sys.exit(1)

    print("Downloading MOT20 from Kaggle (ismailelbouknify/mot-20)…")
    path = kagglehub.dataset_download("ismailelbouknify/mot-20")
    print(f"Downloaded to: {path}")
    return Path(path)


def verify_structure(dest: Path):
    """Print a summary of what was downloaded."""
    train_dir = dest / 'train'
    test_dir  = dest / 'test'
    print(f"\nDataset structure at {dest}:")

    for split_dir in [train_dir, test_dir]:
        if not split_dir.exists():
            print(f"  {split_dir.name}/ — NOT FOUND")
            continue
        seqs = sorted(split_dir.iterdir())
        print(f"\n  {split_dir.name}/")
        for seq in seqs:
            if not seq.is_dir():
                continue
            img_dir = seq / 'img1'
            gt_file = seq / 'gt' / 'gt.txt'
            n_frames = len(list(img_dir.glob('*.jpg'))) if img_dir.exists() else 0
            has_gt   = gt_file.exists()
            print(f"    {seq.name:<15}  {n_frames:5d} frames  gt={'yes' if has_gt else 'no'}")


def main():
    p = argparse.ArgumentParser(description="Download MOT20 dataset")
    p.add_argument('--dest', type=str,
                   default=str(BASE_DIR / 'datasets' / 'MOT20'),
                   help='Destination directory (default: mot/datasets/MOT20)')
    p.add_argument('--skip_download', action='store_true',
                   help='Skip download; just verify existing data')
    args = p.parse_args()

    dest = Path(args.dest)

    if not args.skip_download:
        raw_path = download_kaggle(dest)

        # The kagglehub path may be a versioned sub-directory.
        # Walk up to find the directory containing train/ or MOT20/
        found_mot20 = None
        for candidate in [raw_path, raw_path.parent, raw_path.parent.parent]:
            if (candidate / 'train').exists():
                found_mot20 = candidate
                break
            if (candidate / 'MOT20').exists():
                found_mot20 = candidate / 'MOT20'
                break

        if found_mot20 is None:
            print(f"\nWARNING: Could not auto-detect MOT20 structure in {raw_path}.")
            print(f"Please copy/symlink the MOT20 data to: {dest}")
            print(f"Expected layout:  {dest}/train/MOT20-01/ …")
        elif found_mot20 != dest:
            print(f"Copying/linking {found_mot20} → {dest} …")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(found_mot20, dest, symlinks=True)
            print("Done.")
    else:
        if not dest.exists():
            print(f"ERROR: {dest} does not exist and --skip_download was set.")
            sys.exit(1)

    verify_structure(dest)
    print(f"\nDataset ready at: {dest}")
    print("Next step: python scripts/01_track.py --seq MOT20-01")


if __name__ == '__main__':
    main()
