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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_YAML = os.path.join(BASE_DIR, "datasets", "merged_person_detection", "data.yaml")


def train(device: str):
    model = YOLO("yolo26m.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=40,
        batch=16,
        imgsz=1024,
        patience=10,
        lr0=0.01,
        device=int(device) if device.isdigit() else device,
        project=os.path.join(BASE_DIR, "models"),
        name="yolo26m_ft_v1",
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
        # augmentation
        degrees=2.0,
        translate=0.05,
        scale=0.2,
        mosaic=0.5,
        hsv_h=0.010,
        hsv_s=0.30,
        hsv_v=0.20,
        fliplr=0.5,
        flipud=0.0,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    out = os.path.join(BASE_DIR, "models", "yolo26m_ft_v1", "weights")
    print(f"Best: {out}/best.pt")
    print(f"Last: {out}/last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv26m on person detection")
    parser.add_argument("--device", default="0", help="Device: 0 (CUDA), mps (Apple), cpu")
    args = parser.parse_args()
    train(args.device)
