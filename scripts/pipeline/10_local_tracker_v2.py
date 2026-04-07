"""
10_local_tracker_v2.py — Phase 1 v2: SAM2.1 + D4SM + DINOv2 Tracking Pipeline

Replaces the original YOLO+BoT-SORT+OSNet pipeline with:
  - SAM2.1-Large backbone + D4SM memory management for mask-level tracking
  - YOLO detection every K frames as a proposal generator for new objects
  - DINOv2 ViT-L (1024-D) ReID features from mask-cropped images
  - Mask-based foot-point projection for more accurate ground-plane positions

Processing flow per synchronised frame:
  1. For each camera (sequentially):
     a. If detection frame: run YOLO → match against D4SM tracks → init new
     b. D4SM track step → per-object binary masks
     c. Mark dead objects (invisible > T frames)
     d. Compute mask foot-points → ground-plane projection → filter OOB
     e. Extract DINOv2 features from mask-cropped person images
  2. Save frame JSON + embeddings NPZ (compatible with Phase 2)
  3. Clear GPU cache

Usage:
    python scripts/pipeline/10_local_tracker_v2.py
    python scripts/pipeline/10_local_tracker_v2.py --max_frames 5 --visualize
    python scripts/pipeline/10_local_tracker_v2.py --device cuda

Requires: run setup_tracking_v2.sh first to install SAM2 + download checkpoints
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import torch
from ultralytics import YOLO

from odin_eye.tracking.dam4sam_tracker import (
    D4SMEngine, PerCameraState, mask_foot_point, mask_to_xyxy,
)
from odin_eye.reid.dinov2_extractor import DINOv2ReIDExtractor

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
CAMERA_IDS = ["CAM1"]

GROUND_X_MIN, GROUND_X_MAX = -300.0, 3300.0
GROUND_Y_MIN, GROUND_Y_MAX = -90.0, 1110.0
GROUND_MARGIN = 200.0  # cm

REID_BATCH_SIZE = 64

CAM_COLORS = {
    "C1": (77, 255, 106),  "C2": (255, 210, 0),   "C3": (238, 104, 123),
    "C4": (71, 179, 255),  "C5": (160, 229, 0),    "C6": (180, 105, 255),
    "C7": (235, 206, 135),
}


# ─────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LocalTrackletV2:
    """Atomic tracked-person observation from one camera at one frame."""
    cam_id: str
    obj_id: int              # D4SM object ID (camera-local)
    bbox: list               # [x1, y1, x2, y2] from mask contour
    confidence: float        # YOLO confidence at init, 1.0 for tracked
    mask_area: int           # positive pixel count
    embedding: Optional[np.ndarray] = None   # DINOv2 1024-D
    world_xy: Optional[tuple] = None         # ground-plane (X, Y) in cm
    frame_idx: int = 0


# ─────────────────────────────────────────────────────────────────────
# Ground-plane projection (from mask foot-point)
# ─────────────────────────────────────────────────────────────────────
def project_mask_foot(mask: np.ndarray, H_inv: np.ndarray,
                      margin: float = GROUND_MARGIN):
    """
    Project the foot-point of a segmentation mask to the ground plane.

    Uses the bottom-centre of the mask contour rather than the bbox bottom,
    giving more accurate ground positions especially under occlusion.

    Returns (X, Y) in centimetres or None if out of bounds.
    """
    fp = mask_foot_point(mask)
    if fp is None:
        return None

    u, v = fp
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    world_h = H_inv @ pixel_h

    if abs(world_h[2]) < 1e-10:
        return None

    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]

    if (X < GROUND_X_MIN - margin or X > GROUND_X_MAX + margin or
            Y < GROUND_Y_MIN - margin or Y > GROUND_Y_MAX + margin):
        return None

    return (float(X), float(Y))


def clear_gpu_cache(device_str):
    if 'cuda' in device_str:
        torch.cuda.empty_cache()
    elif 'mps' in device_str:
        torch.mps.empty_cache()
    gc.collect()


# ─────────────────────────────────────────────────────────────────────
# Detection ↔ D4SM matching
# ─────────────────────────────────────────────────────────────────────
def bbox_mask_iou(bbox_xyxy, mask):
    """IoU between a [x1,y1,x2,y2] bbox and a binary mask's bbox."""
    mb = mask_to_xyxy(mask)
    if mb is None:
        return 0.0
    x1 = max(bbox_xyxy[0], mb[0])
    y1 = max(bbox_xyxy[1], mb[1])
    x2 = min(bbox_xyxy[2], mb[2])
    y2 = min(bbox_xyxy[3], mb[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
    area_m = (mb[2] - mb[0]) * (mb[3] - mb[1])
    return inter / max(area_b + area_m - inter, 1e-9)


def match_detections_to_tracks(yolo_bboxes, yolo_confs, d4sm_masks,
                               iou_thresh=0.3):
    """
    Match YOLO bboxes against current D4SM masks.

    Returns:
        matched:    set of YOLO indices that matched an existing track
        unmatched:  list of (bbox, conf) for new objects to initialise
    """
    matched = set()
    if not d4sm_masks:
        return matched, list(zip(yolo_bboxes, yolo_confs))

    obj_ids = list(d4sm_masks.keys())
    for yi, (bbox, _) in enumerate(zip(yolo_bboxes, yolo_confs)):
        best_iou, best_oid = 0.0, None
        for oid in obj_ids:
            iou = bbox_mask_iou(bbox, d4sm_masks[oid])
            if iou > best_iou:
                best_iou, best_oid = iou, oid
        if best_iou >= iou_thresh:
            matched.add(yi)

    unmatched = [(b, c) for i, (b, c) in enumerate(zip(yolo_bboxes, yolo_confs))
                 if i not in matched]
    return matched, unmatched


# ─────────────────────────────────────────────────────────────────────
# Multi-Camera Tracker V2
# ─────────────────────────────────────────────────────────────────────
class MultiCameraTrackerV2:
    def __init__(self, yolo_weights, calib_cache_path, dataset_dir,
                 sam2_checkpoint_dir, sam2_model_size='large',
                 conf_thresh=0.25, imgsz=1280,
                 detect_interval=5, max_lost_frames=10,
                 device='cuda'):
        self.device_str = device
        self.dataset_dir = dataset_dir  # datasets/Wildtrack/Image_subsets
        self.detect_interval = detect_interval
        self.max_lost = max_lost_frames
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Skip calibration (single-camera mode)
        self.calib = None
        self.H_invs = {}

        # Discover synchronised frames
        self.frame_ids = self._discover_frames()
        print(f"Found {len(self.frame_ids)} synchronised frames")

        # YOLO detector (one instance, reset per camera via re-track)
        print(f"Loading YOLO detector ({yolo_weights})...")
        self.yolo = YOLO(yolo_weights)

        # D4SM engine (shared SAM2 backbone)
        self.engine = D4SMEngine(
            model_size=sam2_model_size,
            checkpoint_dir=sam2_checkpoint_dir,
            device=device,
        )

        # Per-camera D4SM states
        self.cam_states: Dict[str, PerCameraState] = {
            cid: PerCameraState(cid) for cid in CAMERA_IDS
        }

        # DINOv2 ReID extractor
        self.reid = DINOv2ReIDExtractor(device=device)

        print(f"MultiCameraTrackerV2 ready — {len(CAMERA_IDS)} cams × "
              f"{len(self.frame_ids)} frames, detect every {detect_interval}f\n")

    # ── frame discovery ───────────────────────────────────────────

    def _discover_frames(self):
        """Discover frame IDs from single-camera folder: dataset_dir/img1/*.jpg"""
        img_dir = os.path.join(self.dataset_dir, "img1")
        exts = (".jpg", ".jpeg", ".png")
        fids = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith(exts)
        )
        return fids

    def _load_image(self, cam_id, frame_id):
        img_dir = os.path.join(self.dataset_dir, "img1")
        candidates = [
            os.path.join(img_dir, f"{frame_id}.jpg"),
            os.path.join(img_dir, f"{frame_id}.jpeg"),
            os.path.join(img_dir, f"{frame_id}.png"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(f"Missing frame: {frame_id} in {img_dir}")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read: {path}")
        return img

    # ── YOLO detection ────────────────────────────────────────────

    def _detect(self, image):
        """Run YOLO on one image, return person bboxes + confidences."""
        results = self.yolo.predict(
            image, conf=self.conf_thresh, classes=[0],
            verbose=False, imgsz=self.imgsz,
        )
        bboxes, confs = [], []
        if results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                bb = results[0].boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                cf = float(results[0].boxes.conf[i].cpu().item())
                bboxes.append(bb)
                confs.append(cf)
        return bboxes, confs

    # ── per-camera frame processing ───────────────────────────────

    def _process_camera(self, cam_id, image_bgr, frame_idx):
        """
        Process one camera for one frame:
          1. (Optional) YOLO detect + match + init new D4SM tracks
          2. D4SM tracking → masks
          3. Kill invisible objects
          4. Ground-plane filter
          5. DINOv2 feature extraction
        Returns list of LocalTrackletV2.
        """
        cs = self.cam_states[cam_id]
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        is_detect_frame = (frame_idx % self.detect_interval == 0)

        # ── Step 1: detection + new object init ──
        if is_detect_frame:
            bboxes, confs = self._detect(image_bgr)

            if not cs.active_ids():
                # No existing tracks: init all detections, use init masks
                if bboxes:
                    _, init_masks = self.engine.initialize_objects(
                        pil, bboxes, cs, frame_idx)
                else:
                    init_masks = {}
                self._kill_lost(cs)
                return self._build_tracklets(
                    cam_id, cs, init_masks, image_bgr, frame_idx)
            else:
                # Track existing objects, then init unmatched detections
                cur_masks = self.engine.track_frame(pil, cs, frame_idx)
                _, unmatched = match_detections_to_tracks(
                    bboxes, confs, cur_masks)
                if unmatched:
                    new_bbs = [bb for bb, _ in unmatched]
                    _, new_masks = self.engine.initialize_objects(
                        pil, new_bbs, cs, frame_idx)
                    cur_masks.update(new_masks)

                self._kill_lost(cs)
                return self._build_tracklets(
                    cam_id, cs, cur_masks, image_bgr, frame_idx)

        # ── Step 2: D4SM tracking (non-detection frames) ──
        masks = self.engine.track_frame(pil, cs, frame_idx)
        self._kill_lost(cs)
        return self._build_tracklets(cam_id, cs, masks, image_bgr, frame_idx)

    def _kill_lost(self, cs):
        for oid in list(cs.alive.keys()):
            if cs.alive[oid] and cs.invisible_count.get(oid, 0) > self.max_lost:
                cs.remove_object(oid)

    # ── tracklet construction + ReID ──────────────────────────────

    def _build_tracklets(self, cam_id, cs, masks, image_bgr, frame_idx):
        """Build LocalTrackletV2 list from D4SM masks, with DINOv2 ReID."""
        tracklets, crops, crop_masks = [], [], []

        for oid, mask in masks.items():
            bbox = mask_to_xyxy(mask)
            if bbox is None:
                continue

            t = LocalTrackletV2(
                cam_id=cam_id,
                obj_id=oid,
                bbox=bbox,
                confidence=1.0,
                mask_area=int(mask.sum()),
                world_xy=None,   # single-camera mode: no ground-plane projection
                frame_idx=frame_idx,
            )
            tracklets.append(t)

            # crop for ReID
            x1, y1, x2, y2 = bbox
            h, w = image_bgr.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = image_bgr[y1c:y2c, x1c:x2c]
            mask_crop = mask[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                crops.append(Image.new("RGB", (10, 10)))
                crop_masks.append(None)
            else:
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                crop_masks.append(mask_crop)

        # batched DINOv2 extraction
        if crops:
            all_emb = []
            for i in range(0, len(crops), REID_BATCH_SIZE):
                batch_c = crops[i:i + REID_BATCH_SIZE]
                batch_m = crop_masks[i:i + REID_BATCH_SIZE]
                emb = self.reid.extract_features_batch(batch_c, masks=batch_m)
                all_emb.append(emb.cpu().numpy())
            embeddings = np.vstack(all_emb) if all_emb else np.empty(
                (0, DINOv2ReIDExtractor.FEATURE_DIM)
            )

            for idx, t in enumerate(tracklets):
                t.embedding = embeddings[idx]

        return tracklets

    # ── full pipeline run ─────────────────────────────────────────

    def run(self, max_frames=None, visualize=False, output_dir="output"):
        frames = self.frame_ids[:max_frames] if max_frames else self.frame_ids
        track_dir = os.path.join(output_dir, "tracking_results_v2")
        os.makedirs(track_dir, exist_ok=True)

        if visualize:
            vis_dir = os.path.join(output_dir, "tracking_vis_v2")
            os.makedirs(vis_dir, exist_ok=True)

        total_tracklets = 0
        t0 = time.time()

        print(f"{'=' * 60}")
        print(f"  Phase 1 v2: SAM2.1 + D4SM + DINOv2 Tracking")
        print(f"  {len(frames)} frames × {len(CAMERA_IDS)} cameras")
        print(f"  detect_interval={self.detect_interval}, "
              f"max_lost={self.max_lost}, device={self.device_str}")
        print(f"{'=' * 60}\n")

        for fi, fid in enumerate(frames):
            ft0 = time.time()
            frame_result: Dict[str, List[LocalTrackletV2]] = {}

            for cam_id in CAMERA_IDS:
                try:
                    img = self._load_image(cam_id, fid)
                    tracklets = self._process_camera(cam_id, img, fi)
                    frame_result[cam_id] = tracklets
                    del img
                except Exception as e:
                    print(f"  WARNING {cam_id} frame {fid}: {e}")
                    frame_result[cam_id] = []

            n = sum(len(v) for v in frame_result.values())
            total_tracklets += n
            cam_str = " | ".join(
                f"{c}:{len(frame_result[c])}" for c in CAMERA_IDS)
            print(f"  Frame {fi + 1:3d}/{len(frames)} [{fid}] — "
                  f"{n:2d} tracklets ({cam_str}) — "
                  f"{time.time() - ft0:.2f}s")

            self._save_frame(fid, fi, frame_result, track_dir)

            if visualize:
                self._visualize(fid, fi, frame_result, vis_dir)

            clear_gpu_cache(self.device_str)

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  Phase 1 v2 Complete!")
        print(f"  Frames: {len(frames)}, Tracklets: {total_tracklets}")
        print(f"  Avg/frame: {elapsed / max(len(frames), 1):.2f}s, "
              f"Total: {elapsed:.1f}s")
        print(f"  Output: {track_dir}/")
        print(f"{'=' * 60}")

    # ── I/O (same JSON+NPZ contract as v1 for Phase 2 compat) ────

    def _save_frame(self, fid, fi, result, out_dir):
        data = {"frame_id": fid, "frame_idx": fi, "cameras": {}}
        npz = {}

        for cam_id, tracklets in result.items():
            data["cameras"][cam_id] = []
            for t in tracklets:
                entry = {
                    "track_id": t.obj_id,
                    "bbox": t.bbox,
                    "confidence": round(t.confidence, 4),
                    "world_xy": list(t.world_xy) if t.world_xy else None,
                    "mask_area": t.mask_area,
                    "embedding_norm": (
                        float(np.linalg.norm(t.embedding))
                        if t.embedding is not None else None),
                }
                data["cameras"][cam_id].append(entry)
                if t.embedding is not None:
                    npz[f"{cam_id}_{t.obj_id}"] = t.embedding

        with open(os.path.join(out_dir, f"frame_{fid}.json"), 'w') as f:
            json.dump(data, f, indent=2)
        if npz:
            np.savez_compressed(
                os.path.join(out_dir, f"frame_{fid}_embeddings.npz"), **npz)

    # ── visualisation ─────────────────────────────────────────────

    def _visualize(self, fid, fi, result, vis_dir):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        images = []
        for cam_id in CAMERA_IDS:
            img = self._load_image(cam_id, fid)
            for t in result.get(cam_id, []):
                x1, y1, x2, y2 = t.bbox
                col = CAM_COLORS[cam_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lbl = f"O{t.obj_id}"
                if t.world_xy:
                    lbl += f" ({t.world_xy[0]:.0f},{t.world_xy[1]:.0f})"
                (tw, th), _ = cv2.getTextSize(
                    lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                cv2.putText(img, lbl, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
            images.append(cv2.resize(img, (640, 360)))
            del img

        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        r1 = np.hstack(images[:4])
        r2 = np.hstack(images[4:] + [blank])
        grid = np.vstack([r1, r2])
        cv2.imwrite(os.path.join(vis_dir, f"grid_{fid}.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])

        # BEV plot
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.add_patch(plt.Rectangle(
            (GROUND_X_MIN, GROUND_Y_MIN), 3600, 1200,
            lw=1.5, ec='#4a8a4a', fc='#1a2a1a', ls='--', alpha=0.5))
        for cam_id, tracklets in result.items():
            bgr = CAM_COLORS[cam_id]
            rgb = f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
            for t in tracklets:
                if t.world_xy is None:
                    continue
                ax.scatter(*t.world_xy, color=rgb, s=80, zorder=5,
                           edgecolors='white', linewidth=0.5)
                ax.text(t.world_xy[0] + 30, t.world_xy[1] + 30,
                        f"{cam_id}:O{t.obj_id}", fontsize=6,
                        color=rgb, alpha=0.8, zorder=6)
        ax.set_xlim(GROUND_X_MIN - 100, GROUND_X_MAX + 100)
        ax.set_ylim(GROUND_Y_MIN - 100, GROUND_Y_MAX + 100)
        ax.set_aspect('equal')
        ax.set_title(f"Frame {fid} — Ground Plane (v2 mask foot-point)",
                     fontsize=12, color='white', fontweight='bold')
        ax.set_xlabel("X (cm)", color='#e0e0e0')
        ax.set_ylabel("Y (cm)", color='#e0e0e0')
        ax.tick_params(colors='#a0a0a0')
        ax.grid(True, alpha=0.15, color='#2a2a4a')
        for sp in ax.spines.values():
            sp.set_color('#4a4a6a')
        plt.savefig(os.path.join(vis_dir, f"bev_{fid}.png"),
                    dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 v2: SAM2.1 + D4SM + DINOv2 Local Tracking")
    p.add_argument("--yolo_weights", default=os.path.join(
        BASE_DIR, "models", "yolo26m_ft_v1", "weights", "best.pt"))
    p.add_argument("--sam2_checkpoint_dir", default=os.path.join(
        BASE_DIR, "checkpoints"))
    p.add_argument("--sam2_model_size", default="large",
                   choices=["large", "base", "small", "tiny"])
    p.add_argument("--calib_cache", default=os.path.join(
        BASE_DIR, "output", "calibration_cache.json"))
    p.add_argument("--dataset_dir", default=os.path.join(
        BASE_DIR, "datasets", "Wildtrack", "Image_subsets"))
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--conf_thresh", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--detect_interval", type=int, default=5,
                   help="Run YOLO every K frames (default: 5)")
    p.add_argument("--max_lost_frames", type=int, default=10,
                   help="Remove object after T invisible frames (default: 10)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--output_dir", default=os.path.join(BASE_DIR, "output"))
    p.add_argument("--device", default="cuda",
                   help="cuda, mps, or cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tracker = MultiCameraTrackerV2(
        yolo_weights=args.yolo_weights,
        calib_cache_path=args.calib_cache,
        dataset_dir=args.dataset_dir,
        sam2_checkpoint_dir=args.sam2_checkpoint_dir,
        sam2_model_size=args.sam2_model_size,
        conf_thresh=args.conf_thresh,
        imgsz=args.imgsz,
        detect_interval=args.detect_interval,
        max_lost_frames=args.max_lost_frames,
        device=args.device,
    )

    tracker.run(
        max_frames=args.max_frames,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )
