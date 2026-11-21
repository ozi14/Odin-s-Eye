"""Frozen table formats shared between modules (ADR-001).

tracks-1.0    : output of M1 (any tracker); input of M2, evaluation and visuals.
gt_tracks-1.0 : MOTChallenge ground truth in the same layout plus the GT-only columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ANCHORS = ("body", "head")
SCHEMA_KEY = b"odins_eye.schema"

_TRACK_FIELDS = [
    pa.field("video_id", pa.string(), nullable=False),
    pa.field("frame", pa.int32(), nullable=False),  # 1-based (MOTChallenge)
    pa.field("track_id", pa.int32(), nullable=False),  # local to one video
    pa.field("x1", pa.float32(), nullable=False),  # pixels, top-left corner
    pa.field("y1", pa.float32(), nullable=False),
    pa.field("x2", pa.float32(), nullable=False),  # pixels, bottom-right corner
    pa.field("y2", pa.float32(), nullable=False),
    pa.field("score", pa.float32(), nullable=False),  # detector confidence; GT = 1.0
    pa.field("anchor", pa.string(), nullable=False),  # "body" | "head"
]

TRACKS_SCHEMA = pa.schema(_TRACK_FIELDS, metadata={SCHEMA_KEY: b"tracks-1.0"})

GT_TRACKS_SCHEMA = pa.schema(
    _TRACK_FIELDS
    + [
        pa.field("consider", pa.bool_(), nullable=False),  # MOT col 7: scored by benchmark
        pa.field("mot_class", pa.int16(), nullable=False),  # MOT col 8: 1 ped, 7 static, ...
        pa.field("visibility", pa.float32(), nullable=False),  # MOT col 9: visible fraction
    ],
    metadata={SCHEMA_KEY: b"gt_tracks-1.0"},
)


class SchemaError(ValueError):
    """A table does not match its frozen format."""


def to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Cast a DataFrame to `schema` (extra columns are dropped) and validate it."""
    missing = [name for name in schema.names if name not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    try:
        table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise SchemaError(f"cannot cast to {schema.metadata[SCHEMA_KEY].decode()}: {exc}") from exc
    table = table.replace_schema_metadata(schema.metadata)
    validate(table, schema)
    return table


def validate(table: pa.Table, schema: pa.Schema) -> None:
    """Raise SchemaError unless `table` has exactly `schema` and sane values."""
    if not table.schema.equals(schema, check_metadata=False):
        raise SchemaError(f"schema mismatch:\n got      {table.schema}\n expected {schema}")
    for name in schema.names:
        if table.column(name).null_count:
            raise SchemaError(f"column {name!r} contains nulls")
    if table.num_rows == 0:
        return
    df = table.select([f.name for f in _TRACK_FIELDS]).to_pandas()
    if (df["frame"] < 1).any():
        raise SchemaError("frame must be >= 1 (frames are 1-based)")
    if ((df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])).any():
        raise SchemaError("every box needs x2 > x1 and y2 > y1")
    if ((df["score"] < 0) | (df["score"] > 1)).any():
        raise SchemaError("score must be in [0, 1]")
    if not df["anchor"].isin(ANCHORS).all():
        raise SchemaError(f"anchor must be one of {ANCHORS}")
    if df.duplicated(["video_id", "frame", "track_id"]).any():
        raise SchemaError("a track_id appears twice in the same frame")


def write_table(df: pd.DataFrame, path: str | Path, schema: pa.Schema) -> Path:
    """Validate and write a table as Parquet; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(df, schema), path)
    return path


def read_table(path: str | Path, schema: pa.Schema) -> pd.DataFrame:
    """Read a Parquet table, check its format version and contents, return a DataFrame."""
    table = pq.read_table(path)
    got = (table.schema.metadata or {}).get(SCHEMA_KEY, b"<none>")
    want = schema.metadata[SCHEMA_KEY]
    if got != want:
        raise SchemaError(f"{path}: format {got.decode()!r}, expected {want.decode()!r}")
    validate(table, schema)
    return table.to_pandas()
