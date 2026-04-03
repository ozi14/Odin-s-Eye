"""
10_local_tracker.py — Phase 1: Per-Camera Local Tracking (Memory-Optimized)

Processes all 7 WILDTRACK cameras SEQUENTIALLY for each synchronized frame:
  1. YOLO detection with custom-trained weights (imgsz=1280 for speed)
  2. BoT-SORT intra-camera tracking (per camera, persist=True)
  3. Ground-plane projection → DISCARD out-of-bounds detections early
  4. Centralized batch ReID extraction (only for in-bounds crops)
  5. Outputs LocalTracklet objects ready for Phase 2 cross-camera association

Memory optimizations vs. v1:
  - Sequential camera processing (not 7× parallel ThreadPool)
  - imgsz 1920 → 1280 (~56% less GPU memory per inference)
  - Early ground-plane filtering: skip ReID for out-of-bounds detections
  - Explicit MPS cache clearing + garbage collection between frames
  - Smaller ReID batches (max 64 crops per batch to cap peak memory)

Usage:
    python legacy/10_local_tracker.py
    python legacy/10_local_tracker.py --split val --max_frames 10
    python legacy/10_local_tracker.py --visualize
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Add project root + legacy (OSNet-AIN) to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "old", "legacy"))

from ultralytics import YOLO
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA_IDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CAM_FILE_MAP = {
    "C1": "cam1", "C2": "cam2", "C3": "cam3", "C4": "cam4",
    "C5": "cam5", "C6": "cam6", "C7": "cam7",
}

# WILDTRACK ground plane bounds (from Phase 0)
GROUND_X_MIN, GROUND_X_MAX = -300.0, 3300.0
GROUND_Y_MIN, GROUND_Y_MAX = -90.0, 1110.0

# Ground-plane margin for accepting detections (cm)
# Detections projecting beyond this are discarded BEFORE ReID
GROUND_MARGIN = 200.0  # 2m margin — keeps edge detections, discards wild ones

# Maximum ReID batch size (caps peak GPU memory usage)
REID_BATCH_SIZE = 64

# Visualization colors per camera (distinct, vibrant — BGR format)
CAM_COLORS = {
    "C1": (77, 255, 106),
    "C2": (255, 210, 0),
    "C3": (238, 104, 123),
    "C4": (71, 179, 255),
    "C5": (160, 229, 0),
    "C6": (180, 105, 255),
    "C7": (235, 206, 135),
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class LocalTracklet:
    """
    A single tracked person observation from one camera at one frame.
    
    This is the atomic unit of data passed from Phase 1 → Phase 2.
    Only contains detections that project INSIDE the ground plane (+margin).
    """
    cam_id: str
    track_id: int
    bbox: list
    confidence: float
    embedding: Optional[np.ndarray] = None
    world_xy: Optional[tuple] = None
    frame_idx: int = 0


# ---------------------------------------------------------------------------
# Ground-Plane Projection
# ---------------------------------------------------------------------------
def project_foot_to_ground(bbox: list, H_inv: np.ndarray,
                            margin: float = GROUND_MARGIN) -> Optional[tuple]:
    """
    Project the foot-point of a bounding box to the ground plane.
    
    The foot-point is the bottom-center of the bbox: ((x1+x2)/2, y2).
    Returns None if the projection falls outside ground bounds + margin.
    
    Args:
        bbox: [x1, y1, x2, y2] in pixels
        H_inv: (3, 3) inverse homography (pixel → world)
        margin: acceptable margin beyond ground plane bounds (cm)
        
    Returns:
        (X, Y) in centimeters on the ground plane, or None if out of bounds
    """
    x1, y1, x2, y2 = bbox
    foot_u = (x1 + x2) / 2.0
    foot_v = float(y2)
    
    pixel_h = np.array([foot_u, foot_v, 1.0], dtype=np.float64)
    world_h = H_inv @ pixel_h
    
    if abs(world_h[2]) < 1e-10:
        return None
    
    X = world_h[0] / world_h[2]
    Y = world_h[1] / world_h[2]
    
    # Strict ground-plane filter: reject detections outside courtyard + margin
    if (X < GROUND_X_MIN - margin or X > GROUND_X_MAX + margin or
        Y < GROUND_Y_MIN - margin or Y > GROUND_Y_MAX + margin):
        return None
    
    return (float(X), float(Y))


def clear_gpu_cache():
    """Release unused GPU memory on MPS backend."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Camera Tracker (one per camera)
# ---------------------------------------------------------------------------
class CameraTracker:
    """
    Manages YOLO + BoT-SORT tracking for a single camera.
    
    Each camera gets its own YOLO model instance because BoT-SORT's
    persist=True maintains internal tracker state that is camera-specific.
    """
    
    def __init__(self, cam_id: str, yolo_weights: str, tracker_config: str,
                 H_inv: np.ndarray, conf_thresh: float = 0.25,
                 imgsz: int = 1280):
        self.cam_id = cam_id
        self.H_inv = H_inv
        self.conf_thresh = conf_thresh
        self.tracker_config = tracker_config
        self.imgsz = imgsz
        
        # Each camera gets its own YOLO instance (required for persist=True)
        self.model = YOLO(yolo_weights)
        
    def process_frame(self, image: np.ndarray, frame_idx: int) -> list:
        """
        Run YOLO + BoT-SORT on one camera frame.
        
        IMPORTANT: Only returns tracklets with valid ground-plane positions.
        Out-of-bounds detections are discarded here (before ReID) to save
        GPU memory and compute.
        
        Returns:
            List of (LocalTracklet, crop_PIL) tuples.
            Embeddings are NOT filled yet — that happens in centralized batch.
        """
        results = self.model.track(
            image,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_thresh,
            classes=[0],
            verbose=False,
            imgsz=self.imgsz,
        )
        
        tracklets_and_crops = []
        discarded = 0
        
        if results[0].boxes is None or results[0].boxes.id is None:
            return tracklets_and_crops
        
        boxes = results[0].boxes
        h, w = image.shape[:2]
        
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            track_id = int(boxes.id[i].cpu().item())
            conf = float(boxes.conf[i].cpu().item())
            
            # ── Early ground-plane filter ──
            # Project BEFORE cropping/ReID to avoid wasting GPU on invalid tracks
            world_xy = project_foot_to_ground(bbox, self.H_inv)
            if world_xy is None:
                discarded += 1
                continue  # Skip this detection entirely
            
            # Crop person from frame
            x1, y1, x2, y2 = bbox
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            
            crop = image[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            tracklet = LocalTracklet(
                cam_id=self.cam_id,
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
                embedding=None,
                world_xy=world_xy,
                frame_idx=frame_idx,
            )
            
            tracklets_and_crops.append((tracklet, crop_pil))
        
        return tracklets_and_crops


# ---------------------------------------------------------------------------
# Multi-Camera Tracker (orchestrator)
# ---------------------------------------------------------------------------
class MultiCameraTracker:
    """
    Orchestrates SEQUENTIAL per-camera tracking across all 7 WILDTRACK cameras.
    
    Processing flow per frame:
      1. Process each camera sequentially (avoids 7× parallel GPU contention)
      2. Discard out-of-ground-plane detections BEFORE ReID
      3. Collect all valid crops, batch ReID in chunks of REID_BATCH_SIZE
      4. Assign embeddings back to tracklets
      5. Clear GPU cache after each frame
    """
    
    def __init__(self, yolo_weights: str, tracker_config: str,
                 calib_cache_path: str, dataset_dir: str,
                 split: str = "val", conf_thresh: float = 0.25,
                 imgsz: int = 1280):
        self.dataset_dir = os.path.join(dataset_dir, split)
        self.split = split
        self.imgsz = imgsz
        
        # Load calibration cache
        print("📦 Loading Phase 0 calibration cache...")
        with open(calib_cache_path, 'r') as f:
            self.calib_cache = json.load(f)
        
        # Discover synchronized frame IDs
        self.frame_ids = self._discover_frames()
        print(f"📁 Found {len(self.frame_ids)} synchronized frames in '{split}' split")
        
        # Initialize per-camera trackers
        print(f"🔧 Initializing {len(CAMERA_IDS)} camera trackers (imgsz={imgsz})...")
        self.trackers = {}
        for cam_id in CAMERA_IDS:
            H_inv = np.array(
                self.calib_cache["cameras"][cam_id]["H_inv"],
                dtype=np.float64
            )
            self.trackers[cam_id] = CameraTracker(
                cam_id=cam_id,
                yolo_weights=yolo_weights,
                tracker_config=tracker_config,
                H_inv=H_inv,
                conf_thresh=conf_thresh,
                imgsz=imgsz,
            )
        
        # Initialize shared ReID extractor
        print("🧠 Loading OSNet-AIN ReID extractor...")
        from osnet_ain.extractor import PersonReIDExtractor
        self.reid_extractor = PersonReIDExtractor()
        
        print(f"✅ MultiCameraTracker ready — {len(CAMERA_IDS)} cameras × "
              f"{len(self.frame_ids)} frames\n")
    
    def _discover_frames(self) -> list:
        """Discover synchronized frame IDs from the dataset directory."""
        files = sorted(os.listdir(self.dataset_dir))
        frame_ids = []
        for f in files:
            if f.startswith("cam1_") and f.endswith(".jpg"):
                fid = f.replace("cam1_", "").replace(".jpg", "")
                frame_ids.append(fid)
        
        verified = []
        for fid in frame_ids:
            all_present = all(
                os.path.exists(os.path.join(self.dataset_dir, f"{CAM_FILE_MAP[cid]}_{fid}.jpg"))
                for cid in CAMERA_IDS
            )
            if all_present:
                verified.append(fid)
        return sorted(verified)
    
    def _load_image(self, cam_id: str, frame_id: str) -> np.ndarray:
        """Load an image for a specific camera and frame."""
        prefix = CAM_FILE_MAP[cam_id]
        path = os.path.join(self.dataset_dir, f"{prefix}_{frame_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to load: {path}")
        return img
    
    def _extract_reid_batched(self, crops: list) -> np.ndarray:
        """
        Extract ReID embeddings in smaller batches to cap peak GPU memory.
        
        Args:
            crops: List of PIL Images
        Returns:
            np.ndarray of shape (N, 512)
        """
        if not crops:
            return np.empty((0, 512))
        
        all_embeddings = []
        for i in range(0, len(crops), REID_BATCH_SIZE):
            batch = crops[i:i + REID_BATCH_SIZE]
            emb = self.reid_extractor.extract_features_batch(batch)
            all_embeddings.append(emb.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def process_frame_set(self, frame_id: str, frame_idx: int) -> dict:
        """
        Process one synchronized frame across all 7 cameras.
        
        SEQUENTIAL processing to avoid GPU memory contention.
        Only in-bounds detections proceed to ReID.
        """
        all_results = {}  # cam_id → [(tracklet, crop), ...]
        total_discarded = 0
        
        # ── Step 1: Sequential YOLO+BoT-SORT per camera ──
        for cam_id in CAMERA_IDS:
            try:
                image = self._load_image(cam_id, frame_id)
                result = self.trackers[cam_id].process_frame(image, frame_idx)
                all_results[cam_id] = result
                del image  # Free frame memory immediately
            except Exception as e:
                print(f"  ⚠️ {cam_id} failed: {e}")
                all_results[cam_id] = []
        
        # ── Step 2: Collect in-bounds crops for ReID ──
        all_crops = []
        crop_indices = []
        
        for cam_id in CAMERA_IDS:
            for i, (tracklet, crop) in enumerate(all_results.get(cam_id, [])):
                all_crops.append(crop)
                crop_indices.append((cam_id, i))
        
        # ── Step 3: Batched ReID extraction ──
        embeddings_np = self._extract_reid_batched(all_crops)
        
        # Free crop references
        del all_crops
        
        # ── Step 4: Assign embeddings back to tracklets ──
        result_dict = {cam_id: [] for cam_id in CAMERA_IDS}
        
        for emb_idx, (cam_id, local_idx) in enumerate(crop_indices):
            tracklet, _ = all_results[cam_id][local_idx]
            tracklet.embedding = embeddings_np[emb_idx]
            result_dict[cam_id].append(tracklet)
        
        # Free intermediate data
        del all_results, crop_indices, embeddings_np
        
        return result_dict
    
    def run(self, max_frames: Optional[int] = None, visualize: bool = False,
            output_dir: str = "output") -> list:
        """Run the full local tracking pipeline on all frames."""
        frames_to_process = self.frame_ids[:max_frames] if max_frames else self.frame_ids
        
        tracking_dir = os.path.join(output_dir, "tracking_results")
        os.makedirs(tracking_dir, exist_ok=True)
        
        if visualize:
            vis_dir = os.path.join(output_dir, "tracking_vis")
            os.makedirs(vis_dir, exist_ok=True)
        
        all_frame_results = []
        total_tracklets = 0
        total_discarded = 0
        total_start = time.time()
        
        print(f"{'='*60}")
        print(f"  Phase 1: Per-Camera Local Tracking (Memory-Optimized)")
        print(f"  {len(frames_to_process)} frames × 7 cameras")
        print(f"  imgsz={self.imgsz}, reid_batch={REID_BATCH_SIZE}, "
              f"ground_margin={GROUND_MARGIN}cm")
        print(f"{'='*60}\n")
        
        for frame_idx, frame_id in enumerate(frames_to_process):
            frame_start = time.time()
            
            # Process all 7 cameras for this frame
            frame_result = self.process_frame_set(frame_id, frame_idx)
            all_frame_results.append(frame_result)
            
            # Count tracklets
            frame_count = sum(len(v) for v in frame_result.values())
            total_tracklets += frame_count
            
            cam_counts = {cid: len(frame_result[cid]) for cid in CAMERA_IDS}
            elapsed = time.time() - frame_start
            
            cam_str = " | ".join(f"{cid}:{cam_counts[cid]}" for cid in CAMERA_IDS)
            print(f"  Frame {frame_idx+1:3d}/{len(frames_to_process)} "
                  f"[{frame_id}] — {frame_count:2d} in-bounds tracks "
                  f"({cam_str}) — {elapsed:.2f}s")
            
            # Visualization (only if requested)
            if visualize:
                self._visualize_frame(frame_id, frame_idx, frame_result, vis_dir)
            
            # Save per-frame JSON
            self._save_frame_json(frame_id, frame_idx, frame_result, tracking_dir)
            
            # ── Memory cleanup every frame ──
            clear_gpu_cache()
        
        # ── Summary ──
        total_elapsed = time.time() - total_start
        avg_per_frame = total_elapsed / len(frames_to_process)
        
        print(f"\n{'='*60}")
        print(f"  Phase 1 Complete!")
        print(f"{'='*60}")
        print(f"  Total frames:        {len(frames_to_process)}")
        print(f"  Total tracklets:     {total_tracklets} (in-bounds only)")
        print(f"  Avg per frame:       {avg_per_frame:.2f}s")
        print(f"  Total time:          {total_elapsed:.1f}s")
        print(f"  Results saved:       {tracking_dir}/")
        if visualize:
            print(f"  Visualizations:      {vis_dir}/")
        print(f"{'='*60}")
        
        self._validate_results(all_frame_results)
        
        return all_frame_results
    
    def _save_frame_json(self, frame_id: str, frame_idx: int,
                          frame_result: dict, output_dir: str):
        """Save one frame's tracklets as JSON and associated embeddings as NPZ."""
        data = {
            "frame_id": frame_id,
            "frame_idx": frame_idx,
            "cameras": {}
        }
        
        npz_data = {}
        
        for cam_id, tracklets in frame_result.items():
            data["cameras"][cam_id] = []
            for t in tracklets:
                entry = {
                    "track_id": t.track_id,
                    "bbox": t.bbox,
                    "confidence": round(t.confidence, 4),
                    "world_xy": list(t.world_xy) if t.world_xy else None,
                    "embedding_norm": float(np.linalg.norm(t.embedding)) if t.embedding is not None else None,
                }
                data["cameras"][cam_id].append(entry)
                
                # Store the actual 512-D embedding for Phase 2
                if t.embedding is not None:
                    # Key format: "C1_5" (Camera C1, Track ID 5)
                    npz_data[f"{cam_id}_{t.track_id}"] = t.embedding
        
        # Save JSON
        json_path = os.path.join(output_dir, f"frame_{frame_id}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Save Embeddings (if any exist)
        if npz_data:
            npz_path = os.path.join(output_dir, f"frame_{frame_id}_embeddings.npz")
            np.savez_compressed(npz_path, **npz_data)
    
    def _visualize_frame(self, frame_id: str, frame_idx: int,
                          frame_result: dict, vis_dir: str):
        """Create multi-camera grid + BEV plot for one frame."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # ── Multi-camera grid ──
        images = []
        for cam_id in CAMERA_IDS:
            img = self._load_image(cam_id, frame_id)
            tracklets = frame_result[cam_id]
            
            for t in tracklets:
                x1, y1, x2, y2 = t.bbox
                color = CAM_COLORS[cam_id]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"T{t.track_id}"
                if t.world_xy:
                    label += f" ({t.world_xy[0]:.0f},{t.world_xy[1]:.0f})"
                
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(img, cam_id, (30, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, CAM_COLORS[cam_id], 3)
            
            img_small = cv2.resize(img, (640, 360))
            images.append(img_small)
            del img  # Free full-res image immediately
        
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        row1 = np.hstack(images[:4])
        row2 = np.hstack(images[4:] + [blank])
        grid = np.vstack([row1, row2])
        del images  # Free resized images
        
        grid_path = os.path.join(vis_dir, f"grid_{frame_id}.jpg")
        cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        del grid
        
        # ── BEV ground-plane plot ──
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        
        ax.add_patch(plt.Rectangle(
            (GROUND_X_MIN, GROUND_Y_MIN), 3600, 1200,
            linewidth=1.5, edgecolor='#4a8a4a', facecolor='#1a2a1a',
            linestyle='--', alpha=0.5
        ))
        
        for cam_id, tracklets in frame_result.items():
            color_bgr = CAM_COLORS[cam_id]
            color_rgb = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            
            for t in tracklets:
                if t.world_xy is None:
                    continue
                X, Y = t.world_xy
                ax.scatter(X, Y, color=color_rgb, s=80, zorder=5,
                          edgecolors='white', linewidth=0.5)
                ax.text(X + 30, Y + 30, f"{cam_id}:T{t.track_id}",
                       fontsize=6, color=color_rgb, alpha=0.8, zorder=6)
        
        ax.set_xlim(GROUND_X_MIN - 100, GROUND_X_MAX + 100)
        ax.set_ylim(GROUND_Y_MIN - 100, GROUND_Y_MAX + 100)
        ax.set_aspect('equal')
        ax.set_title(f"Frame {frame_id} — Ground Plane (in-bounds only)", 
                    fontsize=12, color='white', fontweight='bold')
        ax.set_xlabel("X (cm)", color='#e0e0e0')
        ax.set_ylabel("Y (cm)", color='#e0e0e0')
        ax.tick_params(colors='#a0a0a0')
        ax.grid(True, alpha=0.15, color='#2a2a4a')
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
        
        bev_path = os.path.join(vis_dir, f"bev_{frame_id}.png")
        plt.savefig(bev_path, dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)
        del fig
    
    def _validate_results(self, all_results: list):
        """Run sanity checks on the tracking output."""
        print("\n🔍 Validation Checks:")
        
        total = 0
        no_embedding = 0
        no_world = 0
        in_bounds = 0
        near_edge = 0
        bad_norms = 0
        
        for frame_result in all_results:
            for cam_id, tracklets in frame_result.items():
                for t in tracklets:
                    total += 1
                    
                    if t.embedding is None:
                        no_embedding += 1
                    else:
                        norm = np.linalg.norm(t.embedding)
                        if abs(norm - 1.0) > 0.01:
                            bad_norms += 1
                    
                    if t.world_xy is None:
                        no_world += 1
                    else:
                        X, Y = t.world_xy
                        if (GROUND_X_MIN <= X <= GROUND_X_MAX and
                            GROUND_Y_MIN <= Y <= GROUND_Y_MAX):
                            in_bounds += 1
                        else:
                            near_edge += 1  # Within margin but outside strict bounds
        
        print(f"  Total tracklets:      {total}")
        print(f"  Missing embeddings:   {no_embedding} {'✅' if no_embedding == 0 else '⚠️'}")
        print(f"  Missing world pos:    {no_world} {'✅' if no_world == 0 else '⚠️'}")
        print(f"  Strictly in-bounds:   {in_bounds} ({round(in_bounds/max(total,1)*100,1)}%)")
        print(f"  Near edge (±margin):  {near_edge} ({round(near_edge/max(total,1)*100,1)}%)")
        print(f"  Bad L2 norms:         {bad_norms} {'✅' if bad_norms == 0 else '⚠️'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Per-Camera Local Tracking for WILDTRACK"
    )
    parser.add_argument("--yolo_weights", type=str,
                        default=os.path.join(BASE_DIR, "models", "yolo26_weights_v1", "best.pt"))
    parser.add_argument("--tracker_config", type=str,
                        default=os.path.join(BASE_DIR, "configs", "custom_botsort.yaml"))
    parser.add_argument("--calib_cache", type=str,
                        default=os.path.join(BASE_DIR, "output", "calibration_cache.json"))
    parser.add_argument("--dataset_dir", type=str,
                        default=os.path.join(BASE_DIR, "datasets", "wildtrack", "images"))
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Process only first N frames (for testing)")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="YOLO inference size (default: 1280, was 1920)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images (uses more memory)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "output"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    tracker = MultiCameraTracker(
        yolo_weights=args.yolo_weights,
        tracker_config=args.tracker_config,
        calib_cache_path=args.calib_cache,
        dataset_dir=args.dataset_dir,
        split=args.split,
        conf_thresh=args.conf_thresh,
        imgsz=args.imgsz,
    )
    
    results = tracker.run(
        max_frames=args.max_frames,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )
