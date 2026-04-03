"""
07_train_wildtrack.py — Stage 2: Fine-tune YOLOv26m on WILDTRACK.
Specializes the generalist model to the 7 fixed camera views of the WILDTRACK dataset.
"""

import os
from ultralytics import YOLO

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WILDTRACK_YAML = os.path.join(BASE, "datasets", "wildtrack", "data.yaml")
# Pointing to your Stage 1 'best.pt'
STAGE1_MODEL = os.path.join(BASE, "models", "yolo26m_person_finetune", "weights", "best.pt")
MODEL_DIR = os.path.join(BASE, "models")

def train_stage2():
    # 1. Load the Stage 1 Model
    if os.path.exists(STAGE1_MODEL):
        print(f"✅ Loading Stage 1 model: {STAGE1_MODEL}")
        model = YOLO(STAGE1_MODEL)
    else:
        print(f"⚠️ Stage 1 model not found at {STAGE1_MODEL}. Falling back to pretrained.")
        model = YOLO("yolo26m.pt")

    # 2. Fine-tune on WILDTRACK
    results = model.train(
        data=WILDTRACK_YAML,
        epochs=15,            # Focused adaptation; 15 epochs is usually optimal for 7 fixed views
        batch=8,
        imgsz=832,
        patience=5,
        device="mps",         # Apple M4 Max
        workers=0,
        cache=False,
        project=MODEL_DIR,
        name="yolo26m_wildtrack_finetune",
        save=True,
        plots=True,
    )

    print("\n" + "=" * 50)
    print("STAGE 2 TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best Weights: {MODEL_DIR}/yolo26m_wildtrack_finetune/weights/best.pt")

if __name__ == "__main__":
    train_stage2()
