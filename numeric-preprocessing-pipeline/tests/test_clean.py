"""Stateless cleaning tests: inf→null, domain rules, dedupe, null-target drop."""

from __future__ import annotations

import polars as pl
import pytest

from numprep.clean import clean_stateless
from numprep.config import load_config
from numprep.errors import PipelineError


def test_inf_replaced_with_null() -> None:
    frame = pl.DataFrame({"a": [1.0, float("inf"), -float("inf"), 2.0], "b": ["x", "y", "z", "w"]})
    cfg = load_config()
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned["a"].to_list() == [1.0, None, None, 2.0]
    assert stats.inf_replaced == {"a": 2}
    assert cleaned.columns == ["a", "b"]  # non-numeric columns preserved


def test_inf_to_nan_disabled() -> None:
    frame = pl.DataFrame({"a": [1.0, float("inf")]})
    cfg = load_config()
    cfg.cleaning.inf_to_nan = False
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned["a"].to_list()[1] == float("inf")
    assert stats.inf_replaced == {}


def test_non_negative_domain_rule() -> None:
    frame = pl.DataFrame({"age": [25.0, -3.0, 40.0, None], "b": [1.0, -2.0, 3.0, 4.0]})
    cfg = load_config()
    cfg.cleaning.non_negative_columns = ["age"]
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned["age"].to_list() == [25.0, None, 40.0, None]
    assert stats.negatives_nulled == {"age": 1}
    # Non-configured column untouched.
    assert cleaned["b"].to_list()[1] == -2.0


def test_non_negative_absent_column_raises() -> None:
    frame = pl.DataFrame({"a": [1.0]})
    cfg = load_config()
    cfg.cleaning.non_negative_columns = ["ghost"]
    with pytest.raises(PipelineError, match="ghost"):
        clean_stateless(frame, cfg)


def test_dedupe_removes_exact_duplicates() -> None:
    frame = pl.DataFrame(
        {"a": [1.0, 2.0, 1.0, 2.0], "b": ["x", "y", "x", "y"], "__row_index__": [0, 1, 2, 3]}
    )
    cfg = load_config()
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned.height == 2
    assert stats.duplicates_removed == 2
    # Row-index excluded from the dedupe key: duplicates still collapse.
    assert cleaned["a"].to_list() == [1.0, 2.0]


def test_dedupe_subset_respected() -> None:
    frame = pl.DataFrame({"a": [1.0, 1.0], "b": [1.0, 9.0]})
    cfg = load_config()
    cfg.cleaning.dedupe_subset = ["a"]
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned.height == 1
    assert stats.duplicates_removed == 1
    # keep="first" retains the first occurrence.
    assert cleaned["b"].to_list() == [1.0]


def test_dedupe_subset_absent_column_raises() -> None:
    frame = pl.DataFrame({"a": [1.0]})
    cfg = load_config()
    cfg.cleaning.dedupe_subset = ["ghost"]
    with pytest.raises(PipelineError, match="ghost"):
        clean_stateless(frame, cfg)


def test_dedupe_disabled() -> None:
    frame = pl.DataFrame({"a": [1.0, 1.0]})
    cfg = load_config()
    cfg.cleaning.dedupe = False
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned.height == 2
    assert stats.duplicates_removed == 0


def test_null_target_rows_dropped() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0], "target": [1.0, None, 2.0]})
    cfg = load_config()
    cfg.target.column = "target"
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned.height == 2
    assert stats.null_target_rows_dropped == 1


def test_null_target_rows_kept_when_disabled() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0], "target": [1.0, None]})
    cfg = load_config()
    cfg.target.column = "target"
    cfg.target.drop_null_rows = False
    cleaned, stats = clean_stateless(frame, cfg)
    assert cleaned.height == 2
    assert stats.null_target_rows_dropped == 0


def test_missing_target_column_raises_when_configured() -> None:
    frame = pl.DataFrame({"a": [1.0]})
    cfg = load_config()
    cfg.target.column = "target"
    with pytest.raises(PipelineError, match="target"):
        clean_stateless(frame, cfg)


def test_clean_stats_row_accounting() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 1.0, 2.0, None],
            "target": [1.0, 1.0, None, 3.0],
            "__row_index__": [0, 1, 2, 3],
        }
    )
    cfg = load_config()
    cfg.target.column = "target"
    cleaned, stats = clean_stateless(frame, cfg)
    # dup (1.0/1.0) removed, then the null-target row (2.0/None) removed;
    # the (None, 3.0) row survives (null feature, valid target).
    assert stats.rows_in == 4
    assert stats.rows_out == cleaned.height == 2
