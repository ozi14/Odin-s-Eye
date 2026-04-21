"""
04_train_yolo.py — Fine-tune YOLOv26m on merged person detection dataset.

Single-phase training (CrowdHuman + CCTV Persons).
The resulting weights are used directly by the v2 pipeline.

Usage:
    python scripts/data_prep/04_train_yolo.py
    python scripts/data_prep/04_train_yolo.py --device mps   # Apple Silicon
    python scripts/data_prep/04_train_yolo.py --device 0     # Colab GPU
"""

import os
import argparse
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(BASE_DIR, "data", "merged_person_detection", "data.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
LAST_PT = os.path.join(RUNS_DIR, "yolo26m_ft_v2", "weights", "last.pt")


def train(device: str):
    # Resume from checkpoint if it exists (for HPC auto-resubmit)
    if os.path.exists(LAST_PT):
        print(f"Resuming from {LAST_PT}")
        model = YOLO(LAST_PT)
        do_resume = True
    else:
        print("Starting fresh from yolo26m.pt")
        model = YOLO("yolo26m.pt")
        do_resume = False

    results = model.train(
        data=DATA_YAML,
        epochs=200,
        batch=32,              
        imgsz=1280,
        patience=50,
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,
        device=int(device) if device.isdigit() else device,
        project=RUNS_DIR,
        name="yolo26m_ft_v2",
        save=True,
        save_period=20,
        plots=True,
        verbose=True,
        workers=8,
        resume=do_resume,
        degrees=2.0,
        translate=0.05,
        scale=0.3,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.25,
        fliplr=0.5,
        flipud=0.0,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    out = os.path.join(RUNS_DIR, "yolo26m_ft_v2", "weights")
    print(f"Best: {out}/best.pt")
    print(f"Last: {out}/last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv26m on person detection")
    parser.add_argument("--device", default="0", help="Device: 0 (CUDA), mps (Apple), cpu")
    args = parser.parse_args()
    train(args.device)
