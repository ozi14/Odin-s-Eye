"""
Convert CrowdHuman .odgt annotations to YOLO .txt format.
Uses fbox (full body), skips mask tags and ignored annotations.
"""

import json
import os
from PIL import Image
from tqdm import tqdm

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CH_DIR = os.path.join(BASE, "datasets", "CrowdHuman")


def convert(odgt_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(odgt_path, "r") as f:
        annotations = [json.loads(line.strip()) for line in f]

    stats = {"images": 0, "persons": 0, "skipped_mask": 0, "skipped_ignore": 0, "clipped": 0}

    for ann in tqdm(annotations, desc=f"Converting {os.path.basename(odgt_path)}"):
        img_path = os.path.join(image_dir, f"{ann['ID']}.jpg")
        if not os.path.exists(img_path):
            continue

        img_w, img_h = Image.open(img_path).size
        stats["images"] += 1
        lines = []

        for box in ann["gtboxes"]:
            if box["tag"] != "person":
                stats["skipped_mask"] += 1
                continue
            if box.get("extra", {}).get("ignore", 0) == 1:
                stats["skipped_ignore"] += 1
                continue

            x, y, w, h = box["fbox"]

            # Clip negatives
            if x < 0:
                w += x; x = 0; stats["clipped"] += 1
            if y < 0:
                h += y; y = 0; stats["clipped"] += 1
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w <= 1 or h <= 1:
                continue

            # Normalize to YOLO format
            xc = max(0, min(1, (x + w / 2) / img_w))
            yc = max(0, min(1, (y + h / 2) / img_h))
            wn = max(0, min(1, w / img_w))
            hn = max(0, min(1, h / img_h))

            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            stats["persons"] += 1

        with open(os.path.join(output_dir, f"{ann['ID']}.txt"), "w") as f:
            f.write("\n".join(lines))

    return stats


if __name__ == "__main__":
    print("=" * 50)
    print("CROWDHUMAN → YOLO CONVERSION")
    print("=" * 50)

    for split, odgt, img_dir, lbl_dir in [
        ("TRAIN", "annotation_train.odgt", "Images", "labels_train"),
        ("VAL", "annotation_val.odgt", "Images_val", "labels_val"),
    ]:
        stats = convert(
            os.path.join(CH_DIR, odgt),
            os.path.join(CH_DIR, img_dir),
            os.path.join(CH_DIR, lbl_dir),
        )
        print(f"\n{split}: {stats['images']:,} images, {stats['persons']:,} persons")
        print(f"  Skipped mask: {stats['skipped_mask']:,}, ignore: {stats['skipped_ignore']:,}, clipped: {stats['clipped']:,}")

    print("\n✓ Done")
