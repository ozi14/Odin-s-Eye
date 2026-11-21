# ADR-001: `tracks.parquet` format (tracks-1.0) is frozen

- Status: accepted · 2026-11-20 · owner: Taha
- Code: `src/odins_eye/io/schemas.py` (`TRACKS_SCHEMA`, `GT_TRACKS_SCHEMA`), tested in
  `tests/unit/test_schemas.py`.

## Decision
One row per (video, frame, track):

| column | type | meaning |
|---|---|---|
| video_id | string | sequence name, e.g. `MOT20-01` |
| frame | int32 | 1-based frame index (MOTChallenge) |
| track_id | int32 | identity, local to one video |
| x1, y1, x2, y2 | float32 | box corners in pixels, MOTChallenge coordinates |
| score | float32 | detector confidence in [0, 1]; GT rows use 1.0 |
| anchor | string | `body` or `head` |

Ground truth uses the same columns plus `consider` (bool, MOT column 7), `mot_class`
(int16, MOT column 8: 1 pedestrian, 7 static person, ...) and `visibility` (float32, MOT
column 9). GT is stored **unfiltered**; which rows count is decided downstream
(see the future ADR on GT crowd state).

Every file stores its format version in the Parquet metadata key `odins_eye.schema`.

## Why
Every module after M1 (crowd state, evaluation, visuals) reads this table, so it has to be
stable before teammates code against it. Parquet is typed and much smaller than v3's
one-JSON-per-frame output.

## Alternatives considered
- MOTChallenge `.txt` only: untyped, no video_id or anchor. We still write it, but only for
  TrackEval.
- `x, y, w, h` instead of corners: corners are what detectors output and what IoU and grid
  binning need; conversion happens only at the MOT text boundary.

## Changing it
Any change bumps the version (tracks-1.1, ...) and needs a new ADR agreed by everyone who
reads the table.
