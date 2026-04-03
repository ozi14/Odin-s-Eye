"""
dam4sam_tracker.py — D4SM Tracking Engine (Multi-Camera, No VOT Dependencies)

Port of DAM4SAM-MOT (https://github.com/alanlukezic/d4sm) with:
  - VOT toolkit dependency removed (bbox IoU computed directly)
  - Shared SAM2.1 backbone across multiple PerCameraState objects
  - Mid-video object addition / removal for surveillance use-cases
  - Configurable device (cuda / mps / cpu)

Requires: d4sm's modified SAM2 package (install via setup_tracking_v2.sh)

The core D4SM memory management is preserved exactly:
  - RAM  (Recent Appearance Memory): last N clean frames
  - DRM  (Distractor Resolving Memory): frames with detected distractors
  - Introspection via multi-mask divergence for distractor detection
"""

import os
import logging
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import OrderedDict

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

from sam2.utils.misc import fill_holes_in_mask_scores


# ══════════════════════════════════════════════════════════════════════
#  Utility functions
# ══════════════════════════════════════════════════════════════════════

def bbox_iou_xywh(a, b):
    """IoU between two [x, y, w, h] bounding boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-9)


def keep_largest_component(mask):
    """Keep only the largest connected component of a binary uint8 mask."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 1
    return out


def npmask2box(mask):
    """Binary uint8 mask → [x, y, w, h] bounding box."""
    xs = np.where(mask.sum(0) > 0)[0]
    ys = np.where(mask.sum(1) > 0)[0]
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 1, 1]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1)]


def mask_to_xyxy(mask):
    """Binary uint8 mask → [x1, y1, x2, y2] bounding box."""
    xs = np.where(mask.sum(0) > 0)[0]
    ys = np.where(mask.sum(1) > 0)[0]
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_foot_point(mask):
    """Bottom-center of the mask contour (more accurate than bbox foot)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    max_y = int(ys.max())
    bottom_xs = xs[ys == max_y]
    return (float(bottom_xs.mean()), float(max_y))


# ══════════════════════════════════════════════════════════════════════
#  SAM2 model builder  (adapted from d4sm, no VOT dependency)
# ══════════════════════════════════════════════════════════════════════

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(sd)
    if missing:
        logging.error("Missing keys: %s", missing)
        raise RuntimeError("Checkpoint has missing keys")
    if unexpected:
        logging.error("Unexpected keys: %s", unexpected)
        raise RuntimeError("Checkpoint has unexpected keys")


def _resolve_config_dir():
    """Find the sam2/configs directory inside the installed sam2 package."""
    import sam2 as _sam2
    pkg_dir = os.path.dirname(_sam2.__file__)
    candidates = [
        os.path.join(pkg_dir, "configs"),
        os.path.join(os.path.dirname(pkg_dir), "sam2", "configs"),
        os.path.join(os.path.dirname(pkg_dir), "configs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Cannot find sam2 configs directory. Searched: {candidates}"
    )


def build_sam(config_file, ckpt_path=None, device="cuda"):
    """Build SAM2 model from config + checkpoint, with postprocessing."""
    config_dir = _resolve_config_dir()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    from hydra import initialize_config_dir
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
        ]
        cfg = compose(config_name=config_file, overrides=overrides)

    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    return model


MODEL_CONFIGS = {
    'large':  ("sam2.1_hiera_large.pt",  "sam2.1/sam2.1_hiera_l"),
    'base':   ("sam2.1_hiera_base.pt",   "sam2.1/sam2.1_hiera_b"),
    'small':  ("sam2.1_hiera_small.pt",  "sam2.1/sam2.1_hiera_s"),
    'tiny':   ("sam2.1_hiera_tiny.pt",   "sam2.1/sam2.1_hiera_t"),
}


# ══════════════════════════════════════════════════════════════════════
#  Per-camera tracking state
# ══════════════════════════════════════════════════════════════════════

class PerCameraState:
    """
    Holds all mutable D4SM tracking state for one camera stream.

    Separating state from the engine allows sharing one SAM2 backbone
    across 7 WILDTRACK cameras while keeping per-object memory banks
    independent.
    """
    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.per_obj_mem: dict   = {}   # obj_id → [mem_dict, ...]
        self.per_obj_ptr: dict   = {}   # obj_id → [ptr_dict, ...]
        self.obj_sizes: dict     = {}   # obj_id → [pixel_count, ...]
        self.last_drm_frame: dict = {}  # obj_id → frame_index of last DRM add
        self.drm_deferred: dict  = {}   # obj_id → mem_dict waiting for next frame
        self.all_obj_ids: list   = []
        self.alive: dict         = {}   # obj_id → bool
        self.invisible_count: dict = {} # frames since last visible
        self.next_obj_id: int    = 1

    def active_ids(self):
        return [oid for oid in self.all_obj_ids if self.alive.get(oid, False)]

    def remove_object(self, obj_id):
        self.alive[obj_id] = False
        self.per_obj_mem.pop(obj_id, None)
        self.per_obj_ptr.pop(obj_id, None)
        self.obj_sizes.pop(obj_id, None)
        self.drm_deferred.pop(obj_id, None)
        if obj_id in self.all_obj_ids:
            self.all_obj_ids.remove(obj_id)


# ══════════════════════════════════════════════════════════════════════
#  D4SM Engine  (shared SAM2 backbone + per-camera states)
# ══════════════════════════════════════════════════════════════════════

class D4SMEngine:
    """
    SAM2.1 + D4SM memory management engine.

    One engine instance loads the SAM2 model once.  Multiple cameras
    are tracked via separate PerCameraState objects passed to
    ``initialize_objects`` and ``track_frame``.
    """

    def __init__(self, model_size='large', checkpoint_dir='./checkpoints',
                 device='cuda'):
        ckpt_name, cfg_name = MODEL_CONFIGS[model_size]
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        print(f"Loading SAM2.1-{model_size.title()} on {device}...")
        self.sam = build_sam(cfg_name, ckpt_path, device=device)
        self.device = torch.device(device)

        self.input_size = 1024
        self.fill_hole_area = 8
        self._mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self._std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        self.maskmem_pos_enc = None

        # D4SM hyper-parameters (match the paper)
        self.max_batch = 200
        self.update_delta = 5
        self.max_ram = 3
        self.max_drm = 3
        self.use_last = True

        self._img_w = None
        self._img_h = None
        print("D4SM engine ready.")

    # ── image preprocessing ───────────────────────────────────────

    def _prep(self, pil_image):
        """PIL → (1, 3, 1024, 1024) normalised tensor on device."""
        if self._img_w is None:
            self._img_w, self._img_h = pil_image.size
        arr = np.array(
            pil_image.convert("RGB").resize(
                (self.input_size, self.input_size)
            )
        )
        t = torch.from_numpy(arr / 255.0).permute(2, 0, 1).float()
        t = (t - self._mean) / self._std
        return t.to(self.device).unsqueeze(0)

    def _backbone(self, img_tensor):
        """Shared backbone forward pass → features."""
        out = self.sam.forward_image(img_tensor)
        fpn = out['backbone_fpn']
        vpe = out['vision_pos_enc']
        expanded = {"backbone_fpn": fpn, "vision_pos_enc": vpe}
        _, vf, vp, fs = self.sam._prepare_backbone_features(expanded)
        return vf, vp, fs

    # ── object initialisation ─────────────────────────────────────

    def initialize_objects(self, pil_image, bboxes_xyxy, cam_state,
                           frame_index=0):
        """
        Register new objects in *cam_state* using bounding-box prompts.

        Args:
            pil_image:    PIL Image (original resolution).
            bboxes_xyxy:  List of [x1, y1, x2, y2] in pixel coords.
            cam_state:    PerCameraState for this camera.
            frame_index:  Current frame number.
        Returns:
            List of newly assigned object IDs.
        """
        if not bboxes_xyxy:
            return []

        img = self._prep(pil_image)
        feats, pos, fsz = self._backbone(img)
        new_ids = []

        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = bbox
            box_t = torch.tensor([[x1, y1], [x2, y2]], dtype=torch.float32)
            box_t = box_t.unsqueeze(0)  # (1, 2, 2)
            labels = torch.tensor([[2, 3]], dtype=torch.int32)
            # scale to SAM2 input coords
            box_t = box_t / torch.tensor(
                [pil_image.width, pil_image.height], dtype=torch.float32
            )
            box_t = (box_t * self.input_size).to(self.device)
            labels = labels.to(self.device)

            pt_inputs = {"point_coords": box_t, "point_labels": labels}
            out_dict = {'per_obj_dict': {}, 'maskmem_pos_enc': None}

            cur = self.sam.track_step(
                frame_idx=frame_index,
                is_init_cond_frame=True,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=fsz,
                point_inputs=pt_inputs,
                mask_inputs=None,
                output_dict=out_dict,
                num_frames=None,
                track_in_reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
            )

            pred = cur["pred_masks"]
            if self.fill_hole_area > 0:
                pred = fill_holes_in_mask_scores(pred, self.fill_hole_area)

            hi_res = F.interpolate(
                pred, (self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
            mmf, mpe = self.sam._encode_new_memory(
                current_vision_feats=feats,
                feat_sizes=fsz,
                pred_masks_high_res=hi_res,
                object_score_logits=cur['object_score_logits'],
                is_mask_from_pts=True,
            )
            mmf = mmf.to(torch.bfloat16).to(self.device, non_blocking=True)

            if self.maskmem_pos_enc is None:
                self.maskmem_pos_enc = [x[0:1].clone() for x in mpe]

            oid = cam_state.next_obj_id
            cam_state.per_obj_mem[oid] = [{
                "maskmem_features": mmf,
                "pred_masks": pred,
                "is_init": True,
                "frame_idx": frame_index,
                "is_drm": False,
            }]
            cam_state.per_obj_ptr[oid] = [{
                "obj_ptr": cur["obj_ptr"],
                "frame_idx": frame_index,
                "is_init": True,
            }]
            cam_state.drm_deferred[oid] = None
            cam_state.obj_sizes[oid] = []
            cam_state.last_drm_frame[oid] = -1
            cam_state.alive[oid] = True
            cam_state.invisible_count[oid] = 0
            cam_state.all_obj_ids.append(oid)
            cam_state.next_obj_id += 1
            new_ids.append(oid)

        return new_ids

    # ── per-frame tracking ────────────────────────────────────────

    def track_frame(self, pil_image, cam_state, frame_index):
        """
        Run one D4SM tracking step for all alive objects in *cam_state*.

        Returns:
            dict[obj_id → np.ndarray]  binary masks (H, W) uint8
        """
        active = cam_state.active_ids()
        if not active:
            return {}

        img = self._prep(pil_image)
        feats, pos, fsz = self._backbone(img)

        mpe_dev = (self.maskmem_pos_enc[0].to(self.device)
                   if self.maskmem_pos_enc else None)

        all_dict = {
            'per_obj_dict':     {o: cam_state.per_obj_mem[o] for o in active},
            'per_obj_obj_ptr_dict': {o: cam_state.per_obj_ptr[o] for o in active},
            'maskmem_pos_enc':  mpe_dev,
            'obj_ids_list':     active,
        }

        # batched inference (handles >200 objects gracefully)
        n_runs = ((len(active) - 1) // self.max_batch) + 1
        merged = None

        for r in range(n_runs):
            s = r * self.max_batch
            e = min(len(active), s + self.max_batch)
            batch_ids = active[s:e]

            sub = {
                'per_obj_dict':     {o: all_dict['per_obj_dict'][o] for o in batch_ids},
                'per_obj_obj_ptr_dict': {o: all_dict['per_obj_obj_ptr_dict'][o] for o in batch_ids},
                'maskmem_pos_enc':  mpe_dev,
                'obj_ids_list':     batch_ids,
            }
            out = self.sam.track_step(
                frame_idx=frame_index,
                is_init_cond_frame=False,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=fsz,
                point_inputs=None,
                mask_inputs=None,
                output_dict=sub,
                num_frames=None,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            out['maskmem_pos_enc'] = None

            if merged is None:
                merged = out
            else:
                for k in ('pred_masks', 'obj_ptr', 'object_score_logits',
                          'maskmem_features'):
                    merged[k] = torch.cat([merged[k], out[k]], 0)
                if 'multimasks_logits' in out:
                    merged['multimasks_logits'] = torch.cat(
                        [merged['multimasks_logits'], out['multimasks_logits']], 0)
                if 'ious' in out:
                    merged['ious'] = torch.cat(
                        [merged['ious'], out['ious']], 0)

        # post-process masks to original resolution
        pred_gpu = merged["pred_masks"]
        if self.fill_hole_area > 0:
            pred_gpu = fill_holes_in_mask_scores(pred_gpu, self.fill_hole_area)

        sz = (self._img_h, self._img_w)
        full_masks = F.interpolate(pred_gpu, size=sz, mode="bilinear",
                                   align_corners=False)
        binary = [(m[0] > 0).float().cpu().numpy().astype(np.uint8)
                  for m in full_masks]
        npix = [int(b.sum()) for b in binary]

        mmf = merged["maskmem_features"].to(torch.bfloat16)

        alt_all = F.interpolate(merged["multimasks_logits"], size=sz,
                                mode="bilinear", align_corners=False)
        alt_all = (alt_all > 0).detach().cpu().numpy().astype(np.uint8)
        ious_np = merged["ious"].detach().cpu().numpy()

        # per-object memory update
        for idx, oid in enumerate(active):
            self._update_memory(
                cam_state, oid, idx, frame_index,
                merged, pred_gpu, mmf, binary, npix, alt_all, ious_np,
            )

        return {oid: binary[i]
                for i, oid in enumerate(active)
                if cam_state.alive.get(oid, False)}

    # ── D4SM memory update  (exact paper logic) ──────────────────

    def _update_memory(self, cs, oid, idx, fidx, merged, pred_gpu,
                       mmf, binary, npix, alt_all, ious_np):
        """RAM + DRM update for a single object."""

        # ── deferred DRM from previous frame ──
        if cs.drm_deferred.get(oid):
            mem = cs.per_obj_mem[oid]
            mem[-1] = cs.drm_deferred[oid]
            cs.drm_deferred[oid] = None
            drm_idx = [i for i, e in enumerate(mem)
                       if not e['is_init'] and e['is_drm']]
            if len(drm_idx) > self.max_drm:
                mem.pop(drm_idx[0])

        # ── visibility check ──
        if npix[idx] == 0:
            cs.invisible_count[oid] = cs.invisible_count.get(oid, 0) + 1
            return
        cs.invisible_count[oid] = 0

        mem = cs.per_obj_mem[oid]

        # ── object pointer update ──
        ptr = {
            "obj_ptr": merged["obj_ptr"][idx].unsqueeze(0),
            "frame_idx": fidx,
            "is_init": False,
        }
        ptrs = cs.per_obj_ptr[oid]
        ptrs.append(ptr)
        if len(ptrs) > self.sam.max_obj_ptrs_in_encoder:
            for i, p in enumerate(ptrs):
                if not p["is_init"]:
                    ptrs.pop(i)
                    break

        # ── new memory dict for this frame ──
        new_mem = {
            "maskmem_features": mmf[idx].unsqueeze(0),
            "pred_masks": pred_gpu[idx].unsqueeze(0).detach().cpu().numpy(),
            "is_init": False,
            "frame_idx": fidx,
            "is_drm": False,
        }

        # ── RAM update ──
        if self.use_last:
            ram = [i for i, e in enumerate(mem)
                   if not e['is_init'] and not e['is_drm']]
            if len(ram) == 0:
                mem.append(new_mem)
            elif (fidx % self.update_delta) == 0:
                if (mem[ram[-1]]['frame_idx'] % self.update_delta) == 0:
                    mem.append(new_mem)
                else:
                    mem[ram[-1]] = new_mem
            else:
                if (mem[ram[-1]]['frame_idx'] % self.update_delta) == 0:
                    mem.append(new_mem)
                else:
                    mem[ram[-1]] = new_mem
        else:
            if (fidx % self.update_delta) == 0:
                mem.append(new_mem)

        # evict oldest RAM if over budget
        ram = [i for i, e in enumerate(mem)
               if not e['is_init'] and not e['is_drm']]
        if len(ram) > self.max_ram and len(mem) > self.sam.num_maskmem:
            mem.pop(ram[0])

        # ── DRM update (introspection) ──
        if self.max_drm > 0:
            self._drm_update(cs, oid, idx, fidx, mmf, pred_gpu,
                             binary, npix, alt_all, ious_np)

    def _drm_update(self, cs, oid, idx, fidx, mmf, pred_gpu,
                    binary, npix, alt_all, ious_np):
        """Distractor-aware DRM update via multi-mask introspection."""
        best_i = int(np.argmax(ious_np[idx]))
        best_iou = float(ious_np[idx][best_i])
        alts = [m for i, m in enumerate(alt_all[idx]) if i != best_i]

        # object size stability check
        cs.obj_sizes.setdefault(oid, [])
        cs.obj_sizes[oid].append(npix[idx])
        if len(cs.obj_sizes[oid]) > 1:
            recent = [s for s in cs.obj_sizes[oid][-300:] if s >= 1][-10:]
            ratio = npix[idx] / max(float(np.median(recent)), 1)
        else:
            ratio = -1.0

        last = cs.last_drm_frame.get(oid, -1)

        if not (best_iou > 0.8
                and 0.8 <= ratio <= 1.2
                and (fidx - last > self.update_delta or last == -1)):
            return

        chosen = binary[idx]
        chosen_bb = npmask2box(chosen)

        processed = [np.logical_and(a, ~chosen.astype(bool)).astype(np.uint8)
                     for a in alts]
        processed = [keep_largest_component(p) for p in processed if p.sum() >= 1]
        if not processed:
            return

        unioned = [np.logical_or(p, chosen.astype(bool)).astype(np.uint8)
                   for p in processed]
        alt_bbs = [npmask2box(u) for u in unioned]
        ious = [bbox_iou_xywh(chosen_bb, ab) for ab in alt_bbs]

        if min(ious) > 0.7:
            return

        cs.last_drm_frame[oid] = fidx
        drm_dict = {
            "maskmem_features": mmf[idx].unsqueeze(0),
            "pred_masks": pred_gpu[idx].unsqueeze(0).detach().cpu().numpy(),
            "is_init": False,
            "frame_idx": fidx,
            "is_drm": True,
        }

        mem = cs.per_obj_mem[oid]
        if fidx == mem[-1]['frame_idx']:
            cs.drm_deferred[oid] = drm_dict
        else:
            mem.append(drm_dict)

        # evict if over total memory budget
        if len(mem) > self.sam.num_maskmem:
            drm_idx = [i for i, e in enumerate(mem)
                       if not e['is_init'] and e['is_drm']]
            if len(drm_idx) > self.max_drm:
                mem.pop(drm_idx[0])
            else:
                ram_idx = [i for i, e in enumerate(mem)
                           if not e['is_init'] and not e['is_drm']]
                if ram_idx:
                    mem.pop(ram_idx[0])
