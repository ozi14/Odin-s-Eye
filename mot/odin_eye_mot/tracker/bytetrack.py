"""
bytetrack.py — ByteTrack Dense Pedestrian Tracker

Implements the ByteTrack algorithm (Zhang et al., ECCV 2022) with an
optional DINOv2 ReID secondary association signal.

Two-pass detection association:
  Pass 1: High-confidence detections (≥ thresh_high) → all active+lost
          tracks, using IoU + optional ReID cosine similarity.
  Pass 2: Low-confidence detections (thresh_low–thresh_high) → remaining
          active (not lost) tracks that were unmatched in pass 1.

Track lifecycle:
  - New:      unmatched high-conf detection, confirmed after min_hits frames
  - Tracked:  confirmed, actively matched
  - Lost:     unmatched for ≤ max_lost_frames; eligible for re-association
  - Removed:  lost > max_lost_frames; optionally resurrected via ReID
              within reid_window frames

Usage:
    tracker = ByteTracker(reid_extractor=DINOv2ReIDExtractor())
    for frame in frames:
        detections = yolo_detect(frame)          # list of (bbox_xyxy, score)
        tracks = tracker.update(detections, frame_img)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

from .kalman_filter import KalmanFilter


# ─────────────────────────────────────────────────────────────────────
# Track state
# ─────────────────────────────────────────────────────────────────────

class TrackState(Enum):
    New      = 1   # not yet confirmed
    Tracked  = 2   # actively tracked
    Lost     = 3   # temporarily invisible
    Removed  = 4   # permanently terminated


_kalman = KalmanFilter()   # shared instance (stateless)
_next_id = 1


def _new_id() -> int:
    global _next_id
    _id = _next_id
    _next_id += 1
    return _id


def reset_id_counter():
    """Call between sequences to reset global ID counter."""
    global _next_id
    _next_id = 1


# ─────────────────────────────────────────────────────────────────────
# Track class
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id:    int
    state:       TrackState
    frame_id:    int                      # frame when last seen
    start_frame: int
    hits:        int = 1                  # consecutive matched frames
    time_since_update: int = 0

    # Kalman filter state
    mean:  Optional[np.ndarray] = field(default=None, repr=False)
    cov:   Optional[np.ndarray] = field(default=None, repr=False)

    # Last raw detection
    score:         float = 1.0
    bbox_xyxy:     Optional[np.ndarray] = field(default=None, repr=False)

    # DINOv2 ReID feature gallery (EMA-smoothed)
    reid_feat:     Optional[np.ndarray] = field(default=None, repr=False)
    reid_alpha:    float = 0.9            # EMA smoothing factor

    @classmethod
    def from_detection(cls, bbox_xyxy: np.ndarray, score: float,
                       frame_id: int, reid_feat=None,
                       immediately_confirm: bool = False) -> 'Track':
        tid = _new_id()
        t = cls(
            track_id=tid,
            state=TrackState.Tracked if immediately_confirm else TrackState.New,
            frame_id=frame_id,
            start_frame=frame_id,
            score=score,
            bbox_xyxy=bbox_xyxy.copy(),
            reid_feat=reid_feat,
        )
        meas = _xyxy_to_xywh(bbox_xyxy)
        t.mean, t.cov = _kalman.initiate(meas)
        return t

    # ── properties ──────────────────────────────────────────────────

    @property
    def predicted_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2,
                         cx + w / 2, cy + h / 2], dtype=np.float32)

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.Tracked

    # ── Kalman ops ──────────────────────────────────────────────────

    def predict(self):
        self.mean, self.cov = _kalman.predict(self.mean, self.cov)
        self.time_since_update += 1

    def update(self, bbox_xyxy: np.ndarray, score: float, frame_id: int,
               reid_feat=None):
        meas = _xyxy_to_xywh(bbox_xyxy)
        self.mean, self.cov = _kalman.update(self.mean, self.cov, meas)
        self.bbox_xyxy = bbox_xyxy.copy()
        self.score = score
        self.frame_id = frame_id
        self.hits += 1
        self.time_since_update = 0
        # Any matched track becomes Tracked immediately
        if self.state in (TrackState.New, TrackState.Lost):
            self.state = TrackState.Tracked
        self._update_reid(reid_feat)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def _update_reid(self, feat):
        if feat is None:
            return
        if self.reid_feat is None:
            self.reid_feat = feat.copy()
        else:
            self.reid_feat = (self.reid_alpha * self.reid_feat
                              + (1.0 - self.reid_alpha) * feat)
            norm = np.linalg.norm(self.reid_feat)
            if norm > 1e-9:
                self.reid_feat /= norm

    def to_mot_row(self, frame_id: int) -> list:
        """MOTChallenge format row: [frame, id, x, y, w, h, conf, -1, -1, -1]"""
        x1, y1, x2, y2 = self.predicted_xyxy
        w, h = x2 - x1, y2 - y1
        return [frame_id, self.track_id,
                round(float(x1), 2), round(float(y1), 2),
                round(float(w), 2), round(float(h), 2),
                round(float(self.score), 4), -1, -1, -1]


# ─────────────────────────────────────────────────────────────────────
# IoU utilities
# ─────────────────────────────────────────────────────────────────────

def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    cx = (xyxy[0] + xyxy[2]) / 2.0
    cy = (xyxy[1] + xyxy[3]) / 2.0
    w  = xyxy[2] - xyxy[0]
    h  = xyxy[3] - xyxy[1]
    return np.array([cx, cy, w, h], dtype=np.float64)


def iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bboxes (N×4, M×4) in xyxy format.
    Returns (N, M) IoU matrix.
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0])
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1])
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2])
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = ((bboxes_a[:, 2] - bboxes_a[:, 0])
              * (bboxes_a[:, 3] - bboxes_a[:, 1]))[:, None]
    area_b = ((bboxes_b[:, 2] - bboxes_b[:, 0])
              * (bboxes_b[:, 3] - bboxes_b[:, 1]))[None, :]
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def linear_assignment(cost_matrix: np.ndarray,
                       thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy assignment: iteratively pick minimum-cost entry ≤ thresh.

    Returns:
        matched:   (K, 2) array of (row_idx, col_idx) matched pairs
        unmatched_rows: unmatched row indices
        unmatched_cols: unmatched col indices
    """
    if cost_matrix.size == 0:
        rows = np.arange(cost_matrix.shape[0])
        cols = np.arange(cost_matrix.shape[1])
        return np.empty((0, 2), dtype=int), rows, cols

    # Try scipy for optimal assignment first
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid = cost_matrix[row_ind, col_ind] <= thresh
        matched = np.stack([row_ind[valid], col_ind[valid]], axis=1)
        matched_rows = set(row_ind[valid].tolist())
        matched_cols = set(col_ind[valid].tolist())
    except Exception:
        # Greedy fallback
        cm = cost_matrix.copy()
        matched_rows, matched_cols = set(), set()
        matched_list = []
        while True:
            min_val = cm.min()
            if min_val > thresh:
                break
            ri, ci = np.unravel_index(cm.argmin(), cm.shape)
            matched_list.append([ri, ci])
            matched_rows.add(int(ri))
            matched_cols.add(int(ci))
            cm[ri, :] = 1e9
            cm[:, ci] = 1e9
        matched = (np.array(matched_list, dtype=int)
                   if matched_list else np.empty((0, 2), dtype=int))

    all_rows = set(range(cost_matrix.shape[0]))
    all_cols = set(range(cost_matrix.shape[1]))
    unmatched_rows = np.array(sorted(all_rows - matched_rows), dtype=int)
    unmatched_cols = np.array(sorted(all_cols - matched_cols), dtype=int)
    return matched, unmatched_rows, unmatched_cols


# ─────────────────────────────────────────────────────────────────────
# ByteTracker
# ─────────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    ByteTrack dense multi-object tracker with optional DINOv2 ReID.

    Args:
        thresh_high:      high-confidence detection threshold (pass 1)
        thresh_low:       low-confidence threshold (pass 2 lower bound)
        iou_thresh_high:  IoU cost threshold for pass-1 association
        iou_thresh_low:   IoU cost threshold for pass-2 association
        max_lost_frames:  frames before a Lost track is removed
        min_hits:         consecutive detections before track is confirmed
        reid_extractor:   optional DINOv2ReIDExtractor instance
        reid_thresh:      minimum cosine similarity to accept ReID match
        reid_weight:      blend weight between IoU and ReID cost (0=IoU only)
        reid_window:      frames within which lost tracks can be resurrected
    """

    def __init__(
        self,
        thresh_high:     float = 0.6,
        thresh_low:      float = 0.1,
        iou_thresh_high: float = 0.3,
        iou_thresh_low:  float = 0.5,
        max_lost_frames: int   = 30,
        min_hits:        int   = 1,
        reid_extractor         = None,
        reid_thresh:     float = 0.3,
        reid_weight:     float = 0.3,
        reid_window:     int   = 100,
        reid_batch_size: int   = 64,
    ):
        self.thresh_high     = thresh_high
        self.thresh_low      = thresh_low
        self.iou_thresh_high = iou_thresh_high
        self.iou_thresh_low  = iou_thresh_low
        self.max_lost_frames = max_lost_frames
        self.min_hits        = min_hits
        self.reid            = reid_extractor
        self.reid_thresh     = reid_thresh
        self.reid_weight     = reid_weight
        self.reid_window     = reid_window
        self.reid_batch_size = reid_batch_size

        self.tracked_tracks: List[Track] = []   # active (Tracked/New)
        self.lost_tracks:    List[Track] = []   # Lost
        self.removed_tracks: List[Track] = []   # archive

        self.frame_id = 0

    # ── public API ──────────────────────────────────────────────────

    def update(
        self,
        detections: List[Tuple[np.ndarray, float]],
        frame_img:  Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Process one frame and return currently confirmed active tracks.

        Args:
            detections:  list of (bbox_xyxy, score) from YOLO
            frame_img:   BGR frame image (required for ReID)

        Returns:
            List of active Track objects (state == Tracked)
        """
        self.frame_id += 1

        # ── 0. Extract ReID features for all detections ──────────────
        det_feats = self._extract_reid_features(detections, frame_img)

        # ── 1. Split detections by confidence ────────────────────────
        high_dets = [(b, s, f) for (b, s), f in zip(detections, det_feats)
                     if s >= self.thresh_high]
        low_dets  = [(b, s, f) for (b, s), f in zip(detections, det_feats)
                     if self.thresh_low <= s < self.thresh_high]

        # ── 2. Predict all existing tracks ───────────────────────────
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()

        # ── 3. Separate active tracks by state ───────────────────────
        confirmed   = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        unconfirmed = [t for t in self.tracked_tracks if t.state == TrackState.New]

        # ── 4. Pass 1: high-conf dets → confirmed + lost tracks ──────
        pool = confirmed + self.lost_tracks
        matched1, unmatched_trk1, unmatched_det1 = self._associate(
            high_dets, pool, self.iou_thresh_high)

        for ti, di in matched1:
            b, s, f = high_dets[di]
            pool[ti].update(b, s, self.frame_id, f)

        # unmatched tracks from pass 1 that were active → keep, mark later
        unmatched_confirmed  = [i for i in unmatched_trk1 if i < len(confirmed)]
        unmatched_lost_after1 = [i for i in unmatched_trk1 if i >= len(confirmed)]

        # ── 5. Pass 2: low-conf dets → remaining confirmed tracks ────
        remaining_confirmed = [confirmed[i] for i in unmatched_confirmed]
        matched2, unmatched_trk2, unmatched_det2 = self._associate(
            low_dets, remaining_confirmed, self.iou_thresh_low)

        for ti, di in matched2:
            b, s, f = low_dets[di]
            remaining_confirmed[ti].update(b, s, self.frame_id, f)

        # ── 6. Unconfirmed tracks → high-conf unmatched dets ─────────
        hi_unmatched_dets = [high_dets[i] for i in unmatched_det1]
        matched3, unmatched_unc, unmatched_det3 = self._associate(
            hi_unmatched_dets, unconfirmed, iou_thresh=0.3)

        for ti, di in matched3:
            b, s, f = hi_unmatched_dets[di]
            unconfirmed[ti].update(b, s, self.frame_id, f)

        # ── 7. Mark still-unmatched tracks ───────────────────────────
        still_unmatched_conf = [remaining_confirmed[i]
                                for i in unmatched_trk2]
        for t in still_unmatched_conf:
            t.mark_lost()
        for i in unmatched_unc:
            unconfirmed[i].mark_removed()
        for i in unmatched_lost_after1:
            if i - len(confirmed) >= 0:
                self.lost_tracks[i - len(confirmed)].time_since_update += 0  # already +1'd in predict

        # ── 7b. ReID resurrection: match unmatched high-conf dets to
        #         recently removed tracks via appearance similarity ────
        still_unmatched_det_indices = list(unmatched_det3)
        if (self.reid is not None and self.reid_weight > 0
                and still_unmatched_det_indices and self.removed_tracks):
            # Candidates: removed tracks within reid_window frames
            resurrection_candidates = [
                t for t in self.removed_tracks
                if t.reid_feat is not None
                and (self.frame_id - t.frame_id) <= self.reid_window
            ]
            if resurrection_candidates:
                cand_feats = [t.reid_feat for t in resurrection_candidates]
                det_feats_for_res = [
                    hi_unmatched_dets[i][2] for i in still_unmatched_det_indices
                ]
                # Vectorized cosine similarity
                valid_cand = [i for i, f in enumerate(cand_feats) if f is not None]
                valid_det = [i for i, f in enumerate(det_feats_for_res) if f is not None]
                if valid_cand and valid_det:
                    cand_mat = np.stack([cand_feats[i] for i in valid_cand])
                    det_mat = np.stack([det_feats_for_res[i] for i in valid_det])
                    sim = cand_mat @ det_mat.T  # (n_cand, n_det)
                    resurrected_dets = set()
                    resurrected_cands = set()
                    # Greedy best-match
                    while True:
                        if sim.size == 0:
                            break
                        best_idx = np.unravel_index(sim.argmax(), sim.shape)
                        best_sim = sim[best_idx]
                        if best_sim < self.reid_thresh:
                            break
                        ci, di = best_idx
                        real_ci = valid_cand[ci]
                        real_di = valid_det[di]
                        orig_det_idx = still_unmatched_det_indices[real_di]
                        b, s, f = hi_unmatched_dets[orig_det_idx]
                        # Resurrect: re-activate the old track
                        old_track = resurrection_candidates[real_ci]
                        old_track.update(b, s, self.frame_id, f)
                        old_track.state = TrackState.Tracked
                        self.removed_tracks.remove(old_track)
                        resurrected_dets.add(real_di)
                        resurrected_cands.add(ci)
                        sim[ci, :] = -1
                        sim[:, di] = -1
                    # Update unmatched list
                    still_unmatched_det_indices = [
                        still_unmatched_det_indices[i]
                        for i in range(len(still_unmatched_det_indices))
                        if i not in resurrected_dets
                    ]

        # ── 8. Init new tracks from totally unmatched high-conf dets ─
        # On first frame, immediately confirm all tracks (no min_hits delay)
        is_first_frame = (self.frame_id == 1)
        newly_created = []
        for i in still_unmatched_det_indices:
            b, s, f = hi_unmatched_dets[i]
            if s >= self.thresh_high:
                t = Track.from_detection(b, s, self.frame_id, f,
                                         immediately_confirm=is_first_frame)
                newly_created.append(t)

        # ── 9. Rebuild track lists ────────────────────────────────────
        # Move confirmed-but-now-lost tracks to lost list
        self.lost_tracks = (
            [t for t in pool[len(confirmed):]        # previously lost, still relevant
             if t.state == TrackState.Lost and t.time_since_update <= self.max_lost_frames]
            + still_unmatched_conf
            + [t for t in pool[:len(confirmed)]
               if t.state == TrackState.Lost
               and t not in still_unmatched_conf
               and t.time_since_update <= self.max_lost_frames]
        )
        # Deduplicate
        seen_ids = set()
        unique_lost = []
        for t in self.lost_tracks:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                unique_lost.append(t)
        self.lost_tracks = unique_lost

        # Remove over-age lost tracks
        for t in self.lost_tracks:
            if t.time_since_update > self.max_lost_frames:
                t.mark_removed()
                self.removed_tracks.append(t)
        self.lost_tracks = [t for t in self.lost_tracks
                            if t.state != TrackState.Removed]

        # Active tracks = matched tracked + matched unconfirmed + newly created
        active = (
            [t for t in pool if t.state == TrackState.Tracked]
            + [t for t in unconfirmed if t.state != TrackState.Removed]
            + newly_created
        )
        # Deduplicate
        seen_ids2 = set()
        unique_active = []
        for t in active:
            if t.track_id not in seen_ids2:
                seen_ids2.add(t.track_id)
                unique_active.append(t)
        self.tracked_tracks = unique_active

        return [t for t in self.tracked_tracks
                if t.state == TrackState.Tracked]

    # ── association helpers ──────────────────────────────────────────

    def _associate(
        self,
        dets:     List[Tuple[np.ndarray, float, Optional[np.ndarray]]],
        tracks:   List[Track],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy assignment between detections and tracks.

        Returns (matched_pairs, unmatched_track_indices, unmatched_det_indices).
        matched_pairs: list of (track_idx, det_idx).
        """
        if not dets or not tracks:
            return [], list(range(len(tracks))), list(range(len(dets)))

        pred_boxes = np.array([t.predicted_xyxy for t in tracks], dtype=np.float32)
        det_boxes  = np.array([d[0] for d in dets], dtype=np.float32)
        det_feats  = [d[2] for d in dets]

        ious = iou_matrix(pred_boxes, det_boxes)   # (n_tracks, n_dets)
        cost = 1.0 - ious

        # Blend in ReID cost if available
        if self.reid is not None and self.reid_weight > 0:
            reid_cost = self._reid_cost(tracks, det_feats)
            if reid_cost is not None:
                cost = (1.0 - self.reid_weight) * cost + self.reid_weight * reid_cost

        # Gate: set cost to infinity where IoU is too low
        # iou_thresh is the MINIMUM IoU required for a valid match
        cost[ious < iou_thresh] = 1e9

        matched_arr, unmatched_trk, unmatched_det = linear_assignment(
            cost, thresh=1.0)

        matched_pairs = [(int(r), int(c)) for r, c in matched_arr]
        return matched_pairs, list(unmatched_trk), list(unmatched_det)

    def _reid_cost(self, tracks: List[Track],
                   det_feats: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Cosine distance cost matrix (n_tracks × n_dets), vectorized."""
        trk_feats = [t.reid_feat for t in tracks]
        if all(f is None for f in trk_feats) or all(f is None for f in det_feats):
            return None

        n_t = len(tracks)
        n_d = len(det_feats)
        cost = np.ones((n_t, n_d), dtype=np.float32)

        # Gather valid indices and stack into matrices for vectorized dot product
        valid_t = [i for i, f in enumerate(trk_feats) if f is not None]
        valid_d = [i for i, f in enumerate(det_feats) if f is not None]
        if not valid_t or not valid_d:
            return None

        trk_mat = np.stack([trk_feats[i] for i in valid_t])   # (Nt, D)
        det_mat = np.stack([det_feats[i] for i in valid_d])   # (Nd, D)
        # Features are L2-normalized, so dot product = cosine similarity
        sim = trk_mat @ det_mat.T                              # (Nt, Nd)
        for ri, ti in enumerate(valid_t):
            for ci, di in enumerate(valid_d):
                cost[ti, di] = 1.0 - sim[ri, ci]
        return cost

    # ── ReID feature extraction ──────────────────────────────────────

    def _extract_reid_features(
        self,
        detections: List[Tuple[np.ndarray, float]],
        frame_img:  Optional[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """Extract DINOv2 features for each detection crop."""
        if self.reid is None or frame_img is None or not detections:
            return [None] * len(detections)

        from PIL import Image
        crops = []
        h, w = frame_img.shape[:2]
        for bbox, _ in detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                crops.append(Image.new('RGB', (10, 10)))
            else:
                crop_bgr = frame_img[y1c:y2c, x1c:x2c]
                import cv2
                crops.append(
                    Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))

        feats_list = []
        for i in range(0, len(crops), self.reid_batch_size):
            batch = crops[i:i + self.reid_batch_size]
            feats = self.reid.extract_features_batch(batch)
            feats_list.append(feats.cpu().numpy())

        all_feats = np.vstack(feats_list) if feats_list else np.empty(
            (0, self.reid.FEATURE_DIM))
        return [all_feats[i] for i in range(len(detections))]

    # ── accessors ───────────────────────────────────────────────────

    @property
    def all_active_tracks(self) -> List[Track]:
        return [t for t in self.tracked_tracks
                if t.state in (TrackState.Tracked, TrackState.New)]

    def reset(self):
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        reset_id_counter()
