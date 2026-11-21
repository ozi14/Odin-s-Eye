# Odin's Eye v4: Phase 1 (Measure the crowd)

Offline dense-crowd analysis from a single fixed camera. Phase 1 detects and tracks every
person in MOT20, then turns the tracks into per-second crowd measurements (count, density,
flow, compression), all scored against ground truth.

Design: Design Report v1.2, kept outside the repo in `../Odins_Eye_Design_Report_v1.2_EN.pdf`.
v3 lives on the `v3` branch / `v3.0` tag.

## Setup (Mac, Apple Silicon)

```bash
cd "$HOME/Desktop/Odin's Eye v4/phase1"
uv python install 3.12
uv sync                      # creates .venv from uv.lock
source .venv/bin/activate
pre-commit install           # ruff + safety hooks on every commit
```

## Run

```bash
python scripts/00_prepare_data.py      # MOT20 GT -> data/interim/gt_tracks/MOT20/*.parquet
python scripts/91_e0_validate_eval.py  # E0: validate the evaluator (~2 min) -> experiments/
pytest                                 # unit + integration tests
ruff check . && ruff format --check .
```

E0 needs two reference inputs (not in git):
TrackEval's example data in `data/raw/trackeval_example/` (data.zip from the TrackEval README) and
the original TrackEval in `external/TrackEval` (`git clone https://github.com/JonathonLuiten/TrackEval.git
external/TrackEval && git -C external/TrackEval checkout 12c8791`).

## Layout

```
configs/            Hydra configs (data/, perception/, crowd_state/, experiment/)
src/odins_eye/      io/ (M1 I/O), perception/ (M1), crowd_state/ (M2), eval/, utils/
scripts/            entry points, numbered by pipeline stage
tests/              unit/ and integration/
docs/decisions/     one ADR per design decision
data/               NOT in git: raw/, weights/, interim/, processed/, annotations/
experiments/        NOT in git: one folder per run
```

## Conventions

Frames are 1-indexed (MOTChallenge); fps is read from `seqinfo.ini`, never assumed; track IDs
are local to one video; raw data is read-only; every derived file is produced by a script.
No face recognition, no identification, no matching of people across videos.
