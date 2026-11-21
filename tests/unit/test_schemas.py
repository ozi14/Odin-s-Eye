import pandas as pd
import pytest

from odins_eye.io.schemas import (
    GT_TRACKS_SCHEMA,
    TRACKS_SCHEMA,
    SchemaError,
    read_table,
    to_table,
    write_table,
)


def _tracks(**overrides) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "video_id": ["v1", "v1", "v1"],
            "frame": [1, 1, 2],
            "track_id": [1, 2, 1],
            "x1": [10.0, 50.0, 12.0],
            "y1": [20.0, 20.0, 21.0],
            "x2": [30.0, 70.0, 32.0],
            "y2": [80.0, 90.0, 81.0],
            "score": [0.9, 0.15, 0.8],
            "anchor": ["body"] * 3,
        }
    )
    for key, value in overrides.items():
        df[key] = value
    return df


def test_valid_table_is_cast_to_frozen_types():
    table = to_table(_tracks(), TRACKS_SCHEMA)
    assert table.schema.equals(TRACKS_SCHEMA, check_metadata=True)
    assert str(table.schema.field("frame").type) == "int32"
    assert str(table.schema.field("x1").type) == "float"  # float32


def test_parquet_round_trip_keeps_values_and_version(tmp_path):
    path = write_table(_tracks(), tmp_path / "t.parquet", TRACKS_SCHEMA)
    back = read_table(path, TRACKS_SCHEMA)
    pd.testing.assert_frame_equal(back, _tracks().astype(back.dtypes.to_dict()), check_dtype=True)


def test_reading_with_the_wrong_format_version_fails(tmp_path):
    path = write_table(_tracks(), tmp_path / "t.parquet", TRACKS_SCHEMA)
    with pytest.raises(SchemaError, match="format"):
        read_table(path, GT_TRACKS_SCHEMA)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"frame": [0, 1, 2]}, "1-based"),
        ({"x2": [5.0, 70.0, 32.0]}, "x2 > x1"),
        ({"score": [1.5, 0.1, 0.2]}, "score"),
        ({"anchor": ["feet", "body", "body"]}, "anchor"),
        ({"track_id": [1, 1, 1]}, "twice"),
    ],
)
def test_bad_values_are_rejected(overrides, message):
    with pytest.raises(SchemaError, match=message):
        to_table(_tracks(**overrides), TRACKS_SCHEMA)


def test_missing_column_is_rejected():
    with pytest.raises(SchemaError, match="missing"):
        to_table(_tracks().drop(columns=["anchor"]), TRACKS_SCHEMA)
