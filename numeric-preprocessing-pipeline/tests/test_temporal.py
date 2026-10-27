"""Temporal feature tests: past-only semantics, groups, determinism, errors."""

from __future__ import annotations

import polars as pl
import pytest

from numprep.config import load_config
from numprep.errors import InputContractError, PipelineError
from numprep.temporal import add_temporal_features


def _time_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
            ],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "__row_index__": [0, 1, 2, 3, 4],
        }
    )


def _time_cfg() -> object:
    cfg = load_config()
    cfg.split.strategy = "chronological"
    cfg.split.time_column = "timestamp"
    cfg.cleaning.temporal.enabled = True
    cfg.cleaning.temporal.lags = [1, 2]
    cfg.cleaning.temporal.rolling_windows = [3]
    return cfg


def test_lag_uses_strictly_past_values() -> None:
    cfg = _time_cfg()
    frame, result = add_temporal_features(_time_frame(), cfg)
    assert frame["value__lag1"].to_list() == [None, 10.0, 20.0, 30.0, 40.0]
    assert frame["value__lag2"].to_list() == [None, None, 10.0, 20.0, 30.0]
    assert result.added_columns == ["value__lag1", "value__lag2", "value__rollmean3"]


def test_rolling_window_excludes_current_row() -> None:
    cfg = _time_cfg()
    frame, _ = add_temporal_features(_time_frame(), cfg)
    # rollmean3 at t = mean(value[t-3..t-1]); at t=4 → mean(20,30,40)=30
    assert frame["value__rollmean3"].to_list() == [None, None, None, 20.0, 30.0]


def test_rows_sorted_by_time_before_features() -> None:
    cfg = _time_cfg()
    shuffled = pl.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:02:00",
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
            ],
            "value": [30.0, 10.0, 20.0],
            "__row_index__": [0, 1, 2],
        }
    )
    frame, _ = add_temporal_features(shuffled, cfg)
    assert frame["value"].to_list() == [10.0, 20.0, 30.0]
    assert frame["value__lag1"].to_list()[2] == 20.0


def test_null_time_rows_dropped_and_counted() -> None:
    cfg = _time_cfg()
    frame = _time_frame().with_columns(
        pl.when(pl.arange(0, 5) == 1).then(None).otherwise(pl.col("timestamp")).alias("timestamp")
    )
    frame, result = add_temporal_features(frame, cfg)
    assert result.dropped_null_time_rows == 1
    assert frame.height == 4


def test_grouped_lags_respect_entity_boundaries() -> None:
    cfg = _time_cfg()
    cfg.cleaning.temporal.group_columns = ["entity"]
    frame = pl.DataFrame(
        {
            "timestamp": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
            ],
            "entity": ["a", "a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, 100.0, 200.0],
            "__row_index__": [0, 1, 2, 3, 4],
        }
    )
    out, _ = add_temporal_features(frame, cfg)
    # entity b's first row has no history within its group.
    assert out["value__lag1"].to_list() == [None, 1.0, 2.0, None, 100.0]


def test_auto_columns_exclude_id_and_time() -> None:
    cfg = _time_cfg()
    frame = _time_frame().with_columns(pl.int_range(0, 5).cast(pl.Int64).alias("id"))
    frame, _ = add_temporal_features(frame, cfg)
    added = [c for c in frame.columns if "__lag" in c or "__roll" in c]
    assert added == [c for c in added if not c.startswith("id__")]
    assert any(c.startswith("value__") for c in added)


def test_explicit_columns_validated() -> None:
    cfg = _time_cfg()
    cfg.cleaning.temporal.columns = ["ghost"]
    with pytest.raises(PipelineError, match="ghost"):
        add_temporal_features(_time_frame(), cfg)


def test_missing_time_column_raises() -> None:
    cfg = _time_cfg()
    frame = pl.DataFrame({"value": [1.0], "__row_index__": [0]})
    with pytest.raises(InputContractError, match="timestamp"):
        add_temporal_features(frame, cfg)


def test_missing_group_column_raises() -> None:
    cfg = _time_cfg()
    cfg.cleaning.temporal.group_columns = ["ghost"]
    with pytest.raises(InputContractError, match="ghost"):
        add_temporal_features(_time_frame(), cfg)


def test_temporal_features_deterministic() -> None:
    cfg = _time_cfg()
    out_a, meta_a = add_temporal_features(_time_frame(), cfg)
    out_b, meta_b = add_temporal_features(_time_frame(), cfg)
    assert out_a.equals(out_b)
    assert meta_a == meta_b
