"""
Merge CrowdHuman (YOLO-converted) + CCTV Persons into a single YOLO dataset.
Uses file copies with ch_ / cctv_ prefixes to avoid collisions.
"""

import os
import shutil
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CH_DIR = os.path.join(BASE, "datasets", "CrowdHuman")
CCTV_DIR = os.path.join(BASE, "datasets", "cctv_persons")
MERGED = os.path.join(BASE, "datasets", "merged_person_detection")


def copy_dataset(img_dir, lbl_dir, split, prefix):
    """Copy images + matching labels into merged dataset with prefix."""
    out_img = os.path.join(MERGED, "images", split)
    out_lbl = os.path.join(MERGED, "labels", split)

    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    count = 0

    for img_file in tqdm(images, desc=f"{prefix} → {split}"):
        name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        label_path = os.path.join(lbl_dir, f"{name}.txt")

        if not os.path.exists(label_path):
            continue

        new_name = f"{prefix}_{name}"
        shutil.copy2(os.path.join(img_dir, img_file), os.path.join(out_img, f"{new_name}{ext}"))
        shutil.copy2(label_path, os.path.join(out_lbl, f"{new_name}.txt"))
        count += 1

    return count


if __name__ == "__main__":
    # Create output dirs
    for split in ["train", "val"]:
        os.makedirs(os.path.join(MERGED, "images", split), exist_ok=True)
        os.makedirs(os.path.join(MERGED, "labels", split), exist_ok=True)

    print("=" * 50)
    print("MERGING DATASETS")
    print("=" * 50)

    # CrowdHuman
    ch_train = copy_dataset(
        os.path.join(CH_DIR, "Images"), os.path.join(CH_DIR, "labels_train"), "train", "ch"
    )
    ch_val = copy_dataset(
        os.path.join(CH_DIR, "Images_val"), os.path.join(CH_DIR, "labels_val"), "val", "ch"
    )

    # CCTV Persons
    cctv_train = copy_dataset(
        os.path.join(CCTV_DIR, "train", "images"), os.path.join(CCTV_DIR, "train", "labels"), "train", "cctv"
    )
    cctv_val = copy_dataset(
        os.path.join(CCTV_DIR, "valid", "images"), os.path.join(CCTV_DIR, "valid", "labels"), "val", "cctv"
    )

    # Write data.yaml
    yaml_path = os.path.join(MERGED, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.abspath(os.path.join(MERGED, 'images', 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(MERGED, 'images', 'val'))}\n\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")

    print(f"\nTrain: {ch_train:,} (CrowdHuman) + {cctv_train:,} (CCTV) = {ch_train + cctv_train:,}")
    print(f"Val:   {ch_val:,} (CrowdHuman) + {cctv_val:,} (CCTV) = {ch_val + cctv_val:,}")
    print(f"\n✓ data.yaml: {yaml_path}")
