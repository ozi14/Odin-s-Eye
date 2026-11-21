# ADR-002: Tracking evaluation = TrackEval (validated by E0)

- Status: accepted · 2026-11-20 · owner: Taha
- Code: `src/odins_eye/eval/trackeval_runner.py`; validation `scripts/91_e0_validate_eval.py`

## Decision
- All tracking metrics come from TrackEval through `evaluate_tracks` / `evaluate_mot_files`.
  Package: `trackeval==1.3.0` (maintained PyPI fork, NumPy 2 compatible, same pin as BoxMOT).
- `BENCHMARK` must match the GT (MOT20 adds `non_mot_vehicle` to the distractor classes);
  `DO_PREPROC` stays on, so predictions on distractors are removed rather than counted as FP.
- Main metric HOTA (+ DetA, AssA); also MOTA, IDF1, IDSW, Frag, FP, FN. HOTA/DetA/AssA/LocA are
  means over 19 IoU thresholds; CLEAR metrics use IoU 0.5.
- A metric number may be reported only from code that passes E0.

## Why
v3 used motmetrics with GT filtered to class 1 but unfiltered predictions, so correct boxes on
static people counted as false positives and HOTA was never computed.

## Validation (E0)
A) MPNTrack (TrackEval example tracker) on MOT20-train: our wrapper vs the original
   JonathonLuiten/TrackEval @ 12c8791 (run unmodified in a subprocess with the old NumPy
   aliases restored). B) Known-answer perturbations of GT on all four sequences.
