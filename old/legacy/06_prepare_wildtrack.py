"""
06_prepare_wildtrack.py — Prepare WILDTRACK Multi-Camera Dataset for YOLO training.

Functionality:
1. Downloads WILDTRACK from Kaggle via kagglehub.
2. Converts per-frame JSON annotations to camera-specific YOLO .txt files.
3. Performs a 360-frame train / 40-frame validation split (Total: 400 frames, 7 cameras each).
4. Persists 'calibrations' for downstream Phase 2 (Overlap) and Phase 4 (BEV Fusion).
5. Generates the standard YOLO 'data.yaml' configuration.
"""

import os
import json
import shutil
import logging
import kagglehub
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants (WILDTRACK specific)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE, "datasets", "wildtrack")
CALIB_DIR = os.path.join(BASE, "datasets", "wildtrack_calibrations") # Keep calib separate for logic
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
TOTAL_FRAMES = 400
TRAIN_SPLIT = 360

def convert_to_yolo_format(bbox: List[float], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert [xmin, ymin, xmax, ymax] to normalized [x_center, y_center, width, height].
    Returns None if dimensions are invalid or fully clipped.
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Clip to image boundaries
    xmin = max(0, min(xmin, img_w))
    xmax = max(0, min(xmax, img_w))
    ymin = max(0, min(ymin, img_h))
    ymax = max(0, min(ymax, img_h))

    w = xmax - xmin
    h = ymax - ymin
    
    if w <= 1 or h <= 1: # Discard tiny/invalid boxes
        return None
        
    x_center = xmin + w / 2
    y_center = ymin + h / 2
    
    return (x_center / img_w, y_center / img_h, w / img_w, h / img_h)

def setup_directories():
    """Create output directory structure for YOLO."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)
    os.makedirs(CALIB_DIR, exist_ok=True)

def process_wildtrack():
    # 1. Download Dataset
    logger.info("Downloading WILDTRACK from Kaggle hub...")
    download_raw = kagglehub.dataset_download("aryashah2k/large-scale-multicamera-detection-dataset")
    
    # Locate Wildtrack subfolder (kagglehub might put it deep)
    wildtrack_root = os.path.join(download_raw, "Wildtrack")
    if not os.path.exists(wildtrack_root):
        wildtrack_root = download_raw
        
    img_root = os.path.join(wildtrack_root, "Image_subsets")
    ann_root = os.path.join(wildtrack_root, "annotations_positions")
    raw_calib_root = os.path.join(wildtrack_root, "calibrations")

    # 2. Persist Calibration Data (Phase 2/4 Requirement)
    if os.path.exists(raw_calib_root):
        logger.info(f"Syncing calibrations to {CALIB_DIR}...")
        for item in os.listdir(raw_calib_root):
            s = os.path.join(raw_calib_root, item)
            d = os.path.join(CALIB_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # 3. Process Annotations
    setup_directories()
    json_files = sorted([f for f in os.listdir(ann_root) if f.endswith(".json")])
    if len(json_files) == 0:
        logger.error(f"No JSON files found in {ann_root}")
        return

    splits = {
        "train": json_files[:TRAIN_SPLIT],
        "val": json_files[TRAIN_SPLIT:]
    }
    
    stats = {"images_train": 0, "images_val": 0, "boxes": 0, "skipped": 0}

    for phase, file_list in splits.items():
        logger.info(f"Converting {phase} split...")
        for json_file in tqdm(file_list, desc=phase):
            frame_id = os.path.splitext(json_file)[0]
            with open(os.path.join(ann_root, json_file), 'r') as f:
                data = json.load(f)

            # Dictionary to buffer labels for the 7 cameras (C1 to C7)
            cam_labels = {f"C{i+1}": [] for i in range(7)}
            
            # Handle potential 'root' wrapping in WILDTRACK JSONs
            people = data if isinstance(data, list) else data.get("root", [])

            for person in people:
                for view in person.get("views", []):
                    cam_idx = view.get("viewNum") # 0-6
                    cam_name = f"C{cam_idx + 1}"
                    
                    bbox = [view.get("xmin"), view.get("ymin"), view.get("xmax"), view.get("ymax")]
                    
                    # Skip invisible (-1) or missing boxes
                    if -1 in bbox or None in bbox:
                        stats["skipped"] += 1
                        continue
                    
                    yolo_box = convert_to_yolo_format(bbox, IMG_WIDTH, IMG_HEIGHT)
                    if yolo_box:
                        label_str = f"0 {' '.join(map(str, yolo_box))}"
                        cam_labels[cam_name].append(label_str)
                        stats["boxes"] += 1

            # Save per camera: Cam name + Frame id to prevent collision in flat YOLO dir
            for cam_id in range(1, 8):
                cam_name = f"C{cam_id}"
                src_img_path = os.path.join(img_root, cam_name, f"{frame_id}.png")
                
                # Check for png or jpg (different camera versions vary)
                if not os.path.exists(src_img_path):
                    src_img_path = os.path.join(img_root, cam_name, f"{frame_id}.jpg")
                
                if not os.path.exists(src_img_path):
                    continue

                # Filename example: cam1_00000000.jpg
                filename = f"cam{cam_id}_{frame_id}"
                dst_img = os.path.join(OUTPUT_DIR, "images", phase, f"{filename}.jpg")
                dst_lbl = os.path.join(OUTPUT_DIR, "labels", phase, f"{filename}.txt")

                shutil.copy2(src_img_path, dst_img)
                with open(dst_lbl, 'w') as f_lbl:
                    f_lbl.write("\n".join(cam_labels[cam_name]))
                
                stats[f"images_{phase}"] += 1

    # 4. Generate data.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")

    logger.info("=" * 30)
    logger.info("PREPARATION COMPLETE")
    logger.info(f"Processed: {stats['images_train']:,} Train / {stats['images_val']:,} Val Images")
    logger.info(f"Instances: {stats['boxes']:,} boxes created")
    logger.info(f"Skipped:   {stats['skipped']:,} invisible annotations")
    logger.info(f"Calib:     Saved to {CALIB_DIR}")
    logger.info(f"Config:    {yaml_path}")
    logger.info("=" * 30)

if __name__ == "__main__":
    process_wildtrack()
