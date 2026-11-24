# Odin's Eye — Phase 1: Measuring Dense Crowds

**Multi-object tracking and crowd-state estimation from a single camera.**
CS 5330 (PRCV) term project · Northeastern University, Khoury College · Advisor: Dr. Akram Bayat

We detect and track every person in dense-crowd video from one fixed camera (MOT20), turn the
tracks into per-second crowd measurements (count, density, flow, compression), and score both
against ground truth. Offline only. No face recognition, no identification of individuals.

**Research questions**
- **Q1** How does the accuracy of modern trackers change as crowd density increases?
- **Q2** How accurately can crowd state be measured from predicted tracks, compared with ground-truth tracks?
- **Q3** Where and why do tracking and measurement fail?

> **Scope of this repository:** Phase 1 only (the term project, presented 10–13 December 2026).
> Later phases (grounded narration, benchmark, incident triage) will live in a separate repository.
> The previous system (v3: ByteTrack + DINOv2 + Qwen3-VL) is archived on the `v3` branch / `v3.0` tag.

---

## Team and workstreams

| Person | Workstream | Responsibilities | Main folders |
|---|---|---|---|
| **Taha Oğuzhan Uçar** (lead) | Perception + crowd state + integration | Detector and trackers (M1) on Explorer HPC, crowd-state encoder (M2), frozen data formats, code review, final integration | `src/odins_eye/perception/`, `src/odins_eye/crowd_state/`, `src/odins_eye/io/` |
| **Jinsha Lu** | Evaluation | Tracking metrics on all four MOT20 sequences (E1), results by crowd density, crowd-state error tables (E2), failure-analysis statistics (Q3) | `src/odins_eye/eval/`, `scripts/9x_*.py` |
| **Shourya Kumar Srivastava** | Infrastructure + visuals | CI (ruff + pytest on every PR), Hydra configs, SLURM job templates, density/flow heatmaps over video, demo video | `.github/`, `configs/`, `slurm/`, `src/odins_eye/viz/` |

Taha may also contribute to any workstream. Every workstream can start **now** without waiting for
the tracker: use the ground truth as fake predictions (see [Working before the tracker exists](#working-before-the-tracker-exists)).

---

## Pipeline and status

```
video ─► M1 Perception (detect + track) ─► tracks.parquet ─► M2 Crowd state ─► crowd_state.parquet
                         │                                         │
                         └──────────► Evaluation (TrackEval, crowd-state error) ◄───┘
```

| # | Step | Output | Owner | Status |
|---|---|---|---|---|
| 0 | Repo skeleton + environment | layout, frozen `tracks-1.0` format (ADR-001) | Taha | ✅ |
| 1 | Data I/O | MOT20 GT → `gt_tracks` parquet, streaming frame reader | Taha | ✅ |
| 2 | **E0** evaluator validation | TrackEval wrapper validated (ADR-002), 92/92 checks | Taha | ✅ |
| 3 | Detection | YOLO26m detections, conf 0.1, all 4 sequences ([results](docs/results/step3/README.md)) | Taha | ✅ done (recall 81–87%) |
| 4 | Tracking | BoxMOT trackers + fixed v3 ByteTrack → `tracks.parquet` | Taha | |
| 5 | **E1** tracker comparison | HOTA / IDF1 / IDSW tables by density | Jinsha | |
| 6 | M2 crowd state | `crowd_state.parquet` (16×16 grid, 12 features per cell, 1-s windows) | Taha | |
| 7 | **E2** crowd-state accuracy | predicted vs GT crowd-state error by density | Jinsha | |
| 8 | Visuals + demo | heatmaps, flow overlays, demo video, failure examples | Shourya | |

Milestones: code freeze **6 Dec** · results freeze **7 Dec** · presentation **10–13 Dec**.

---

## Setup

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), and git.

```bash
git clone https://github.com/ozi14/Odin-s-Eye.git odins-eye && cd odins-eye
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv is missing
uv python install 3.12
uv sync                     # creates .venv exactly from uv.lock
source .venv/bin/activate
pre-commit install          # ruff + safety checks before every commit
pytest                      # should pass; data-dependent tests skip if data is missing
```

**Troubleshooting (macOS):** `ModuleNotFoundError: No module named 'odins_eye'` while the venv is
active means the `.pth` files in `.venv` got the macOS *hidden* flag (Python 3.12.13 skips hidden
`.pth` files). Fix: `chflags -R nohidden .venv`.

Add a dependency with `uv add <pkg>` (dev-only: `uv add --dev <pkg>`). Never `pip install` into
`.venv`, or teammates end up with different environments. After every `git pull`: `uv sync`.

### Data (not in git)

```
data/
├── raw/MOT20/{train,test}/MOT20-XX/   download from https://motchallenge.net/data/MOT20/
├── raw/trackeval_example/             only for E0: data.zip linked in the TrackEval README
├── weights/                           detector weights: ask Taha (not needed for eval/visuals)
└── interim/gt_tracks/MOT20/           created by scripts/00_prepare_data.py
```

```bash
python scripts/00_prepare_data.py       # MOT20 GT -> data/interim/gt_tracks/MOT20/*.parquet
python scripts/91_e0_validate_eval.py   # optional: re-validate the evaluator (~2 min)
```

E0 also needs the original TrackEval for its reference check:
`git clone https://github.com/JonathonLuiten/TrackEval.git external/TrackEval && git -C external/TrackEval checkout 12c8791`.

Detections and predicted tracks (large) are shared by Taha as parquet files; each PR that
needs them says where to put them under `data/interim/` or `data/processed/`.

---

## Data contracts (read these first)

Modules talk to each other only through these tables. They are **frozen**: a change needs a
version bump and a new decision record agreed by everyone who reads the table.

**`tracks-1.0`**: one row per (video, frame, track). Code: `src/odins_eye/io/schemas.py`;
rationale: [`docs/decisions/ADR-001-tracks-format.md`](docs/decisions/ADR-001-tracks-format.md).

| column | type | meaning |
|---|---|---|
| `video_id` | string | e.g. `MOT20-01` |
| `frame` | int32 | 1-based frame index |
| `track_id` | int32 | identity, local to one video |
| `x1, y1, x2, y2` | float32 | box corners in pixels |
| `score` | float32 | detector confidence in [0, 1] (GT = 1.0) |
| `anchor` | string | `body` or `head` |

**`gt_tracks-1.0`**: the same columns plus `consider` (bool), `mot_class` (int16; 1 pedestrian,
7 static person, ...) and `visibility` (float32). Stored unfiltered.

**`crowd_state-1.0`**: defined in step 6 (ADR-003).

```python
from odins_eye.io.schemas import TRACKS_SCHEMA, GT_TRACKS_SCHEMA, read_table, write_table

gt = read_table("data/interim/gt_tracks/MOT20/MOT20-01.parquet", GT_TRACKS_SCHEMA)
write_table(
    my_tracks_df, "data/interim/tracks/mytracker/MOT20-01.parquet", TRACKS_SCHEMA
)  # validates
```

---

## Working before the tracker exists

Ground truth converted to `tracks-1.0` behaves exactly like a perfect tracker, and the
perturbations in `odins_eye.eval.sanity` turn it into imperfect trackers with known answers:

```python
from odins_eye.io.mot import read_mot_gt
from odins_eye.eval.sanity import scored_gt, drop_rows, swap_ids
from odins_eye.eval.trackeval_runner import evaluate_tracks

gt = read_mot_gt("data/raw/MOT20/train/MOT20-01")
fake = drop_rows(scored_gt(gt), frac=0.2)  # misses 20% of boxes
res = evaluate_tracks({"MOT20-01": fake}, "data/raw/MOT20/train", "experiments/scratch", "fake")
print(res[["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]])
```

### Jinsha (evaluation): first tasks
1. Read `src/odins_eye/eval/trackeval_runner.py`; run `scripts/91_e0_validate_eval.py`.
2. `scripts/90_evaluate.py`: evaluate a folder of `tracks-1.0` files on all four sequences and save
   the results table to `experiments/<date>_e1_<desc>/` (use `odins_eye.utils.run.new_run_dir`).
3. Density breakdown: per-frame bins **and per-cell (local) density** bins. MOT20 has only 46–195
   pedestrians per frame on average, so per-frame bins above ~200 stay empty.
4. Rules: report only numbers produced through `evaluate_tracks` (validated by E0). Never filter
   predictions by class; TrackEval removes predictions on distractors itself.

### Shourya (infrastructure + visuals): first tasks
1. `.github/workflows/ci.yml`: `uv sync` → `ruff check .` → `ruff format --check .` → `pytest` on
   every PR. Integration tests skip automatically without MOT20, so CI needs no dataset.
2. `slurm/`: an Explorer `.sbatch` template requesting `--gres=gpu:a100:1` or `gpu:h200:1`
   (not V100/P100: our torch build uses CUDA 13, which does not support them).
3. `src/odins_eye/viz/`: draw `tracks-1.0` boxes + IDs on frames (stream with
   `odins_eye.io.frames.iter_frames`), then density heatmaps from GT tracks, then a demo-video writer.

---

## How we work

### Branches

```
feat/<stream>-<topic>  ──PR──►  main  ◄──PR (each finished step)──  v4-phase1  (Taha)
                                  │                                     ▲
                                  └────── git merge origin/main ────────┘
```

| Branch | Who | Purpose |
|---|---|---|
| `main` | everyone, **only through PRs** | shared, always-working term-project code |
| `v4-phase1` | Taha | lead's development branch; merged into `main` by PR at each finished step (at least weekly) |
| `feat/…`, `fix/…`, `exp/…`, `docs/…` | Jinsha, Shourya | one short-lived branch per task, created from `main` |
| `v3` (+ tag `v3.0`) | nobody | archived previous system |

`main` is protected: no direct pushes (admins included), no force-push, no deletion. A PR is
required but needs 0 approvals, so the author can merge once checks pass. PRs are merged with a
**merge commit** only (squash/rebase are disabled, so long-lived branches keep identical commits).

### Teammates: one task = one branch = one PR

```bash
git switch main && git pull --ff-only
git switch -c feat/eval-density-bins          # name: <type>/<stream>-<topic>
# ... work, commit, push as often as you like ...
git push -u origin feat/eval-density-bins
git fetch origin && git merge origin/main     # before opening the PR: include the latest main
gh pr create --base main --title "feat(eval): density-binned E1 results" --body "What / why / how tested"
```
Every push to the branch updates the open PR. After the merge, delete the branch and start the
next task from a fresh `main`.

### Taha: develop on `v4-phase1`, merge into `main` by PR

```bash
git switch v4-phase1 && git push                     # day-to-day work
gh pr create --base main --head v4-phase1 --title "Step N: …" --body "…"
gh pr merge --merge                                  # merge commit on main
git fetch origin && git merge --ff-only origin/main && git push   # keep v4-phase1 == main
```
After a teammate's PR is merged: `git fetch origin && git merge origin/main && git push` on
`v4-phase1`, so the lead branch has their code.

### Rules

- **Small and often:** PRs under ~400 changed lines; merge into `main` at least weekly. Long-lived
  differences between branches turn into merge conflicts.
- **Stay in your folders** (see the team table); shared files (`io/schemas.py`, `pyproject.toml`,
  configs) change only with a short note in the PR.
- **Before every PR:** `ruff check . && ruff format . && pytest` pass locally (CI will enforce this).
- **Never** `git push --force` on `main` or `v4-phase1`; never commit data, weights, videos or
  experiment outputs (pre-commit blocks files over 1 MB).
- **Commits:** Conventional Commits: `feat(eval): ...`, `fix(io): ...`, `docs: ...`, `test: ...`.
- **Frozen formats** (`tracks-1.0`, `detections-1.0`) change only with a new ADR and a version bump.
- **Experiments:** every reported number comes from a script that writes
  `experiments/YYYYMMDD_<exp>_<desc>/` with the resolved config, git hash, metrics and logs.
  Commit before running, so `git.json` records a clean hash. Small result summaries go to
  `docs/results/<step>/`.
- **Decisions:** one short record per decision in `docs/decisions/ADR-NNN-*.md`.
- **Sync:** 30-minute weekly meeting; shared results log updated every Friday.

### Daily commands

```bash
source .venv/bin/activate
pytest                                   # all tests
ruff check . && ruff format .            # lint + format
python scripts/00_prepare_data.py        # rebuild GT parquet
python scripts/91_e0_validate_eval.py    # E0
```

---

## Repository layout

```
configs/                Hydra configs: config.yaml, data/mot20.yaml, e0.yaml
src/odins_eye/
├── io/                 frozen formats (schemas), MOT read/write, seqinfo, streaming frames
├── perception/         M1: detector wrapper, tracker adapters, interpolation      (steps 3-4)
├── crowd_state/        M2: grid features from tracks                               (step 6)
├── eval/               TrackEval wrapper, known-answer sanity suite, crowd-state metrics
└── utils/              paths, experiment folders
scripts/                entry points, numbered by stage (00_, 10_, 20_, 9x_ for evaluation)
tests/                  unit/ (no data needed) and integration/ (skipped without MOT20)
docs/decisions/         ADR-001 tracks format, ADR-002 evaluation, ADR-003 detector
docs/results/           per-step results (tables, small figures, run metadata)
data/                   NOT in git (see above)
experiments/            NOT in git: one folder per run
external/               NOT in git: reference third-party code (TrackEval for E0)
```

## Conventions

Pixels and seconds (no homography for MOT20) · frames are 1-indexed · fps is read from
`seqinfo.ini`, never assumed · track IDs are local to one video · raw data is read-only · every
derived file is produced by a script · no face crops saved, no matching of people across videos.

## References

MOT20 (Dendorfer et al., 2020) · HOTA / TrackEval (Luiten et al., IJCV 2021) · ByteTrack (Zhang et al., ECCV 2022) ·
OC-SORT (Cao et al., CVPR 2023) · BoT-SORT (Aharon et al., 2022) · Deep OC-SORT (Maggiolino et al., 2023) ·
BoxMOT (Broström) · YOLO26 (Ultralytics, 2026) · CrowdHuman (Shao et al., 2018) · CroHD (Sundararaman et al., CVPR 2021).
