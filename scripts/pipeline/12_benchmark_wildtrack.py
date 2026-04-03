"""
12_benchmark_wildtrack.py — Evaluate Odin's Eye vs WILDTRACK Ground Truth

Metrics:
  1. Per-Camera Detection: Precision / Recall / F1 via IoU bbox matching
  2. Ground-Plane MODA/MODP: Re-projects GT foot-points through OUR homographies
     so both GT and predictions live in the same coordinate system
  3. Cross-Camera Identity Consistency: checks if same GT person gets
     the same Global ID across all cameras they appear in

Usage:
    python scripts/pipeline/12_benchmark_wildtrack.py
    python scripts/pipeline/12_benchmark_wildtrack.py --iou_thresh 0.4
"""

import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

CAMERA_IDS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def project_foot(H_inv, bbox):
    """Project foot-point of bbox through inverse homography to ground plane."""
    u = (bbox[0]+bbox[2])/2.0
    v = float(bbox[3])
    p = np.array([u, v, 1.0], dtype=np.float64)
    w = H_inv @ p
    if abs(w[2]) < 1e-10:
        return None
    return (w[0]/w[2], w[1]/w[2])


class WildtrackEvaluator:
    def __init__(self, gt_dir, local_dir, global_dir, calib_path,
                 iou_thresh=0.5, dist_thresh_cm=100.0):
        self.gt_dir = gt_dir
        self.local_dir = local_dir
        self.global_dir = global_dir
        self.iou_thresh = iou_thresh
        self.dist_thresh = dist_thresh_cm

        # Load calibration for ground-plane re-projection
        with open(calib_path) as f:
            calib = json.load(f)
        self.H_invs = {}
        for cid in CAMERA_IDS:
            self.H_invs[cid] = np.array(calib["cameras"][cid]["H_inv"], dtype=np.float64)

        # Discover prediction frames
        pred_files = sorted(glob.glob(os.path.join(local_dir, "frame_*.json")))
        self.frame_ids = []
        for f in pred_files:
            fname = os.path.basename(f)
            if "_embeddings" not in fname:
                self.frame_ids.append(fname.replace("frame_","").replace(".json",""))

        print(f"📊 Evaluator: {len(self.frame_ids)} frames, IoU={iou_thresh}, "
              f"dist={dist_thresh_cm}cm\n")

    # ── Loaders ────────────────────────────────────────────────
    def _load_gt(self, frame_id):
        path = os.path.join(self.gt_dir, f"{frame_id}.json")
        if not os.path.exists(path): return []
        with open(path) as f: return json.load(f)

    def _load_local(self, frame_id):
        path = os.path.join(self.local_dir, f"frame_{frame_id}.json")
        if not os.path.exists(path): return {}
        with open(path) as f: return json.load(f)

    def _load_global(self, frame_id):
        path = os.path.join(self.global_dir, f"frame_{frame_id}_global.json")
        if not os.path.exists(path): return {"global_tracks":[], "cameras":{}}
        with open(path) as f: return json.load(f)

    # ── 1. Per-Camera Detection ────────────────────────────────
    def eval_per_camera(self):
        stats = {c: {"tp":0,"fp":0,"fn":0} for c in CAMERA_IDS}
        for fid in self.frame_ids:
            gt_raw = self._load_gt(fid)
            pred = self._load_local(fid)
            for cam_id in CAMERA_IDS:
                view_num = int(cam_id[1]) - 1
                gt_boxes = []
                for p in gt_raw:
                    v = p["views"][view_num]
                    if v["xmin"] != -1:
                        gt_boxes.append([v["xmin"], v["ymin"], v["xmax"], v["ymax"]])
                pred_boxes = [t["bbox"] for t in pred.get("cameras",{}).get(cam_id,[])]

                matched_gt, matched_pred = set(), set()
                if gt_boxes and pred_boxes:
                    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
                    for i, gb in enumerate(gt_boxes):
                        for j, pb in enumerate(pred_boxes):
                            iou_mat[i,j] = compute_iou(gb, pb)
                    while True:
                        best = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                        if iou_mat[best] < self.iou_thresh: break
                        matched_gt.add(best[0]); matched_pred.add(best[1])
                        iou_mat[best[0],:] = 0; iou_mat[:,best[1]] = 0

                stats[cam_id]["tp"] += len(matched_gt)
                stats[cam_id]["fp"] += len(pred_boxes) - len(matched_pred)
                stats[cam_id]["fn"] += len(gt_boxes) - len(matched_gt)
        return stats

    # ── 2. Ground-Plane MODA/MODP ──────────────────────────────
    def eval_ground_plane(self):
        """
        Re-projects GT person foot-points through OUR homographies so both
        GT and predictions live in the same coordinate frame.
        Uses the AVERAGE ground position across all visible cameras as GT.
        """
        total_tp = total_fp = total_fn = 0
        total_dist = 0.0; total_matched = 0

        for fid in self.frame_ids:
            gt_raw = self._load_gt(fid)
            glob_data = self._load_global(fid)

            # GT: project each person's foot-point from ALL visible cameras, average
            gt_positions = []
            for p in gt_raw:
                projections = []
                for v in p["views"]:
                    if v["xmin"] == -1: continue
                    cam_id = f"C{v['viewNum']+1}"
                    bbox = [v["xmin"], v["ymin"], v["xmax"], v["ymax"]]
                    pt = project_foot(self.H_invs[cam_id], bbox)
                    if pt: projections.append(pt)
                if projections:
                    avg_x = np.mean([px for px,py in projections])
                    avg_y = np.mean([py for px,py in projections])
                    gt_positions.append((avg_x, avg_y))

            # Pred: global track positions
            pred_positions = []
            for g in glob_data.get("global_tracks", []):
                if g.get("world_xy"):
                    pred_positions.append(tuple(g["world_xy"]))

            # Distance-based greedy matching
            m_gt, m_pred = set(), set()
            if gt_positions and pred_positions:
                dm = np.zeros((len(gt_positions), len(pred_positions)))
                for i, gp in enumerate(gt_positions):
                    for j, pp in enumerate(pred_positions):
                        dm[i,j] = np.sqrt((gp[0]-pp[0])**2 + (gp[1]-pp[1])**2)
                while True:
                    best = np.unravel_index(np.argmin(dm), dm.shape)
                    if dm[best] > self.dist_thresh: break
                    m_gt.add(best[0]); m_pred.add(best[1])
                    total_dist += dm[best]; total_matched += 1
                    dm[best[0],:] = 1e9; dm[:,best[1]] = 1e9

            total_tp += len(m_gt)
            total_fp += len(pred_positions) - len(m_pred)
            total_fn += len(gt_positions) - len(m_gt)

        return {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "avg_dist": total_dist / max(total_matched, 1),
            "matched": total_matched
        }

    # ── 3. Identity Consistency ────────────────────────────────
    def eval_identity(self):
        total_pairs = correct_pairs = 0
        for fid in self.frame_ids:
            gt_raw = self._load_gt(fid)
            glob_data = self._load_global(fid)

            pred_lookup = defaultdict(list)
            for cam_id in CAMERA_IDS:
                for e in glob_data.get("cameras",{}).get(cam_id,[]):
                    pred_lookup[cam_id].append(e)

            for p in gt_raw:
                visible = [(f"C{v['viewNum']+1}", [v["xmin"],v["ymin"],v["xmax"],v["ymax"]])
                           for v in p["views"] if v["xmin"] != -1]
                if len(visible) < 2: continue

                matched_gids = []
                for cam_id, gt_bbox in visible:
                    best_iou, best_gid = 0, None
                    for pred in pred_lookup[cam_id]:
                        iou = compute_iou(gt_bbox, pred["bbox"])
                        if iou > best_iou:
                            best_iou = iou; best_gid = pred["global_id"]
                    if best_iou >= self.iou_thresh and best_gid:
                        matched_gids.append(best_gid)

                if len(matched_gids) >= 2:
                    n_pairs = len(matched_gids)*(len(matched_gids)-1)//2
                    total_pairs += n_pairs
                    if len(set(matched_gids)) == 1:
                        correct_pairs += n_pairs
        return {
            "total_pairs": total_pairs,
            "correct": correct_pairs,
            "consistency": correct_pairs / max(total_pairs, 1)
        }

    # ── Full Report ────────────────────────────────────────────
    def run(self):
        print(f"{'='*70}")
        print("  WILDTRACK Benchmark — Odin's Eye Pipeline")
        print(f"{'='*70}\n")

        # 1. Per-Camera
        print("📹 Per-Camera Detection (IoU ≥ {:.1f})".format(self.iou_thresh))
        print(f"   {'Cam':<6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"   {'-'*46}")
        cs = self.eval_per_camera()
        ttp = tfp = tfn = 0
        for c in CAMERA_IDS:
            tp,fp,fn = cs[c]["tp"],cs[c]["fp"],cs[c]["fn"]
            ttp+=tp; tfp+=fp; tfn+=fn
            pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1)
            f1 = 2*pr*rc/max(pr+rc,1e-9)
            print(f"   {c:<6} {tp:6} {fp:6} {fn:6} {pr:7.3f} {rc:7.3f} {f1:7.3f}")
        pr = ttp/max(ttp+tfp,1); rc = ttp/max(ttp+tfn,1)
        f1 = 2*pr*rc/max(pr+rc,1e-9)
        print(f"   {'-'*46}")
        print(f"   {'ALL':<6} {ttp:6} {tfp:6} {tfn:6} {pr:7.3f} {rc:7.3f} {f1:7.3f}")

        # 2. Ground-Plane
        print(f"\n🗺️  Ground-Plane Detection (dist ≤ {self.dist_thresh}cm)")
        gp = self.eval_ground_plane()
        gt_total = gp["tp"]+gp["fn"]
        moda = 1.0 - (gp["fn"]+gp["fp"])/max(gt_total,1)
        modp = 1.0 - gp["avg_dist"]/self.dist_thresh if gp["matched"]>0 else 0
        gp_pr = gp["tp"]/max(gp["tp"]+gp["fp"],1)
        gp_rc = gp["tp"]/max(gp["tp"]+gp["fn"],1)
        print(f"   GT people:    {gt_total}")
        print(f"   Predicted:    {gp['tp']+gp['fp']}")
        print(f"   TP/FP/FN:     {gp['tp']} / {gp['fp']} / {gp['fn']}")
        print(f"   Precision:    {gp_pr:.3f}")
        print(f"   Recall:       {gp_rc:.3f}")
        print(f"   Avg dist:     {gp['avg_dist']:.1f} cm")
        print(f"   ★ MODA:       {moda:.3f}")
        print(f"   ★ MODP:       {modp:.3f}")

        # 3. Identity
        print(f"\n🆔 Cross-Camera Identity Consistency")
        ids = self.eval_identity()
        print(f"   Total pairs:  {ids['total_pairs']}")
        print(f"   Correct:      {ids['correct']}")
        print(f"   ★ Consistency: {ids['consistency']:.3f}")

        print(f"\n{'='*70}")
        print("  Evaluation Complete")
        print(f"{'='*70}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", default=os.path.join(BASE_DIR,"datasets","Wildtrack","annotations_positions"))
    p.add_argument("--local_dir", default=os.path.join(BASE_DIR,"output","tracking_results"))
    p.add_argument("--global_dir", default=os.path.join(BASE_DIR,"output","global_results"))
    p.add_argument("--calib", default=os.path.join(BASE_DIR,"output","calibration_cache.json"))
    p.add_argument("--iou_thresh", type=float, default=0.5)
    p.add_argument("--dist_thresh", type=float, default=100.0)
    a = p.parse_args()

    WildtrackEvaluator(a.gt_dir, a.local_dir, a.global_dir, a.calib,
                       a.iou_thresh, a.dist_thresh).run()
