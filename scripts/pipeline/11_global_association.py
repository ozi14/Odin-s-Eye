"""
11_global_association.py — Phase 2: Global Cross-Camera Identity Association

Loads local intra-camera tracklets and their ReID embeddings.  Uses a centralized
Hungarian matching algorithm to fuse overlapping local tracks into persistent
Global Identities (e.g. "G001") based on spatial proximity on the ground plane
AND visual appearance vectors.

Supports both pipeline versions:
  v1: 10_local_tracker.py   → OSNet-AIN 512-D  → tracking_results/
  v2: 10_local_tracker_v2.py → DINOv2   1024-D → tracking_results_v2/

The embedding dimension is auto-detected from the first NPZ file.
Gate thresholds are tuned for DINOv2 but degrade gracefully for OSNet.

Outputs:
  - Global tracklet JSON files
"""

import os
import sys
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# ---------------------------------------------------------------------------
# Constants & Hyperparameters
# ---------------------------------------------------------------------------
CAMERA_IDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CAM_FILE_MAP = {
    "C1": "cam1", "C2": "cam2", "C3": "cam3", "C4": "cam4",
    "C5": "cam5", "C6": "cam6", "C7": "cam7",
}

# Matching weights — tuned for DINOv2 1024-D features.
# DINOv2 cross-view cosine similarity for same person is ~0.65-0.80
# (vs OSNet-AIN ~0.41), so we can trust appearance more.
WEIGHT_REID = 0.50
WEIGHT_SPATIAL = 0.50

# Spatial gate — tighter than v1 because mask-based foot-points are
# more accurate than bbox bottom-centre, reducing the 3.7m projection
# error observed with OSNet bboxes under bottom-occlusion.
MAX_SPATIAL_GATE = 300.0  # cm (was 400)

# Appearance gate — tighter since DINOv2 features are more discriminative.
# Same person across views rarely exceeds 0.40 cosine distance with DINOv2.
MAX_APPEARANCE_COST = 0.45  # (was 0.60)

# EMA threshold for adaptive embedding merge — raised because DINOv2
# provides high similarity for same-person observations.
EMA_MERGE_THRESH = 0.70  # (was 0.65)

EMA_REID = 0.15     # slightly faster adaptation since DINOv2 is reliable
EMA_SPATIAL = 0.5

MAX_LOST_FRAMES = 5

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class LocalObservation:
    """Incoming local detection from a single camera in a single frame."""
    cam_id: str
    track_id: int
    bbox: list
    confidence: float
    world_xy: tuple
    embedding: np.ndarray

class GlobalIdentity:
    """A tracked person existing in the global world map."""
    def __init__(self, global_id: str, init_obs: LocalObservation, frame_idx: int):
        self.global_id = global_id
        
        # State
        self.latest_embedding = init_obs.embedding.copy()
        self.latest_world_xy = init_obs.world_xy
        
        # Tracking metadata
        self.last_seen_frame = frame_idx
        self.frames_lost = 0
        self.is_active = True
        
        # Current frame association
        self.current_observations = {} # cam_id -> dict of LocalObservation info
        self._add_observation(init_obs)

    def _add_observation(self, obs: LocalObservation):
        self.current_observations[obs.cam_id] = {
            "bbox": obs.bbox,
            "local_track_id": obs.track_id,
            "confidence": obs.confidence
        }
        
    def update(self, observations: List[LocalObservation], frame_idx: int):
        """Update global state with new associated observations."""
        self.current_observations = {}
        if not observations:
            self.frames_lost += 1
            if self.frames_lost >= MAX_LOST_FRAMES:
                self.is_active = False
            return
            
        self.last_seen_frame = frame_idx
        self.frames_lost = 0
        
        # Average new spatial and embedding observations (if multiple cameras see them)
        avg_emb = np.zeros_like(self.latest_embedding)
        valid_emb_count = 0
        avg_x, avg_y = 0.0, 0.0
        
        for obs in observations:
            self._add_observation(obs)
            avg_x += obs.world_xy[0]
            avg_y += obs.world_xy[1]
            
            cos_sim = np.dot(self.latest_embedding, obs.embedding)
            if cos_sim > EMA_MERGE_THRESH:
                avg_emb += obs.embedding
                valid_emb_count += 1
            
        avg_x /= len(observations)
        avg_y /= len(observations)
        
        # EMA Spatial Updates (Always happens if assigned)
        old_x, old_y = self.latest_world_xy
        new_x = (1.0 - EMA_SPATIAL) * old_x + EMA_SPATIAL * avg_x
        new_y = (1.0 - EMA_SPATIAL) * old_y + EMA_SPATIAL * avg_y
        self.latest_world_xy = (float(new_x), float(new_y))
        
        # EMA Appearance Updates (Only if good vectors were found)
        if valid_emb_count > 0:
            avg_emb = avg_emb / valid_emb_count
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            self.latest_embedding = (1.0 - EMA_REID) * self.latest_embedding + EMA_REID * avg_emb
            self.latest_embedding /= np.linalg.norm(self.latest_embedding) # Re-normalize


# ---------------------------------------------------------------------------
# Hungarian Matching Pipeline
# ---------------------------------------------------------------------------
class GlobalAssociationTracker:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.active_tracks: Dict[str, GlobalIdentity] = {}
        self.next_global_id = 1
        
        # Find all JSON frames (exclude embedding NPZs that also match frame_*)
        self.json_files = sorted(
            f for f in glob.glob(os.path.join(input_dir, "frame_*.json"))
            if "_embeddings" not in f
        )
        
    def _load_frame_data(self, json_path: str) -> List[LocalObservation]:
        """Load JSON tracklets and their NPZ embeddings for a given frame."""
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        frame_id = data["frame_id"]
        npz_path = os.path.join(self.input_dir, f"frame_{frame_id}_embeddings.npz")
        
        embeddings = {}
        if os.path.exists(npz_path):
            with np.load(npz_path) as npz:
                embeddings = {k: npz[k] for k in npz.files}

        observations = []
        for cam_id, tracklets in data["cameras"].items():
            for t in tracklets:
                if t["world_xy"] is None:
                    continue

                key = f"{cam_id}_{t['track_id']}"
                if key not in embeddings:
                    continue

                obs = LocalObservation(
                    cam_id=cam_id,
                    track_id=t["track_id"],
                    bbox=t["bbox"],
                    confidence=t["confidence"],
                    world_xy=tuple(t["world_xy"]),
                    embedding=embeddings[key],
                )
                observations.append(obs)

        return observations, data

    def _compute_cost_matrix(self, globals_list: List[GlobalIdentity], 
                             obs_list: List[LocalObservation]) -> np.ndarray:
        """Compute spatial+appearance cost matrix between active tracks and incoming observations."""
        n_tracks = len(globals_list)
        n_obs = len(obs_list)
        
        if n_tracks == 0 or n_obs == 0:
            return np.empty((0, 0))
            
        cost_matrix = np.zeros((n_tracks, n_obs), dtype=np.float32)
        
        for i, g_trk in enumerate(globals_list):
            gx, gy = g_trk.latest_world_xy
            g_emb = g_trk.latest_embedding
            
            for j, o_obs in enumerate(obs_list):
                ox, oy = o_obs.world_xy
                o_emb = o_obs.embedding
                
                # 1. Spatial Cost (Euclidean distance on ground)
                spatial_dist = np.sqrt((gx - ox)**2 + (gy - oy)**2)
                
                if spatial_dist > MAX_SPATIAL_GATE:
                    cost_matrix[i, j] = 1e5
                    continue
                    
                # Normalize spatial dist [0, 1] relative to gate
                norm_spatial_cost = spatial_dist / MAX_SPATIAL_GATE
                
                # 2. Appearance Cost (Cosine distance)
                cosine_sim = np.dot(g_emb, o_emb)
                app_cost = 1.0 - cosine_sim
                
                if app_cost > MAX_APPEARANCE_COST:
                    cost_matrix[i, j] = 1e5
                    continue
                
                # Total Fused Cost
                cost_matrix[i, j] = (WEIGHT_REID * app_cost) + (WEIGHT_SPATIAL * norm_spatial_cost)
                
        return cost_matrix

    def process_frame(self, json_path: str, frame_idx: int):
        """Run Global Association for a single frame."""
        obs_list, original_data = self._load_frame_data(json_path)
        frame_id = original_data["frame_id"]
        
        # Extract active tracks
        active_list = [t for t in self.active_tracks.values() if t.is_active]
        
        if not active_list and obs_list:
            # First frame or all tracks died — spawn everything
            for obs in obs_list:
                gid = f"G{self.next_global_id:03d}"
                self.active_tracks[gid] = GlobalIdentity(gid, obs, frame_idx)
                self.next_global_id += 1
        elif active_list and obs_list:
            # Match PER CAMERA to allow one Global Identity to exist in multiple cameras simultaneously
            assignments = {g.global_id: [] for g in active_list}
            assigned_obs = set()
            
            for cam_id in CAMERA_IDS:
                cam_obs = [obs for obs in obs_list if obs.cam_id == cam_id]
                if not cam_obs:
                    continue
                    
                cost_matrix = self._compute_cost_matrix(active_list, cam_obs)
                row_inds, col_inds = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_inds, col_inds):
                    if cost_matrix[r, c] < 1e4:
                        g_trk = active_list[r]
                        obs = cam_obs[c]
                        assignments[g_trk.global_id].append(obs)
                        assigned_obs.add((obs.cam_id, obs.track_id))
            
            # Update all active tracks (even unassigned ones to increment lost counter)
            for g_trk in active_list:
                g_trk.update(assignments[g_trk.global_id], frame_idx)
                
            # Spawn new global identities for unassigned observations
            for obs in obs_list:
                if (obs.cam_id, obs.track_id) not in assigned_obs:
                    gid = f"G{self.next_global_id:03d}"
                    self.active_tracks[gid] = GlobalIdentity(gid, obs, frame_idx)
                    self.next_global_id += 1
        else:
            # No observations in frame, age all active tracks
            for g_trk in active_list:
                g_trk.update([], frame_idx)
                
        # --- Clean dead tracks ---
        to_delete = [gid for gid, t in self.active_tracks.items() if not t.is_active]
        for gid in to_delete:
            del self.active_tracks[gid]
            
        # --- Export Global JSON ---
        self._export_global_json(original_data, frame_idx, frame_id)

    def _export_global_json(self, original_data: dict, frame_idx: int, frame_id: str):
        """Inject global IDs into the tracking result format."""
        global_data = {
            "frame_id": frame_id,
            "frame_idx": frame_idx,
            "global_tracks": [],
            "cameras": {cam: [] for cam in CAMERA_IDS}
        }
        
        # Write Global Summary First
        for g_trk in self.active_tracks.values():
            if not g_trk.is_active or not g_trk.current_observations:
                continue
                
            global_data["global_tracks"].append({
                "global_id": g_trk.global_id,
                "world_xy": g_trk.latest_world_xy,
                "cameras_present": list(g_trk.current_observations.keys())
            })
            
            # Populate per-camera lists
            for cam_id, obs_dict in g_trk.current_observations.items():
                global_data["cameras"][cam_id].append({
                    "global_id": g_trk.global_id,
                    "local_track_id": obs_dict["local_track_id"],
                    "bbox": obs_dict["bbox"],
                    "confidence": obs_dict["confidence"]
                })
                
        out_path = os.path.join(self.output_dir, f"frame_{frame_id}_global.json")
        with open(out_path, 'w') as f:
            json.dump(global_data, f, indent=2)

    def run(self):
        print(f"{'='*60}")
        print("  Phase 2: Global Cross-Camera Identity Association")
        print(f"  Loaded {len(self.json_files)} synchronized frames.")
        print(f"{'='*60}\n")
        
        start_t = time.time()
        for idx, json_path in enumerate(self.json_files):
            self.process_frame(json_path, idx)
            print(f"  Processed frame {idx+1:3d}/{len(self.json_files)} | "
                  f"Active Global IDs: {len([t for t in self.active_tracks.values() if t.is_active])}")
                  
        total_t = time.time() - start_t
        print(f"\n{'='*60}")
        print(f"  Phase 2 Complete in {total_t:.2f}s!")
        print(f"  Output saved to: {self.output_dir}")
        print(f"  Total unique identities detected: {self.next_global_id - 1}")
        print(f"{'='*60}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Cross-Camera Global Association")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Override input directory (auto-selects v2 if present)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--v1", action="store_true",
                        help="Force v1 paths (tracking_results / global_results)")

    args = parser.parse_args()

    if args.input_dir is None:
        v2 = os.path.join(BASE_DIR, "output", "tracking_results_v2")
        v1 = os.path.join(BASE_DIR, "output", "tracking_results")
        args.input_dir = v1 if args.v1 else (v2 if os.path.isdir(v2) else v1)
    if args.output_dir is None:
        suffix = "global_results" if args.v1 else "global_results_v2"
        args.output_dir = os.path.join(BASE_DIR, "output", suffix)

    os.makedirs(args.output_dir, exist_ok=True)
    tracker = GlobalAssociationTracker(args.input_dir, args.output_dir)
    tracker.run()
