"""Audit tests: statistics correctness, flags, sampling labels, report rendering."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest

from numprep.audit import audit_frame
from numprep.config import load_config


@pytest.fixture()
def audit_frame_fixture(rng_gen: np.random.Generator) -> pl.DataFrame:
    n = 400
    x = rng_gen.normal(0.0, 1.0, n)
    nulls = [None if rng_gen.random() < 0.4 else 1.0 for _ in range(n)]
    frame = pl.DataFrame(
        {
            "id": np.arange(n),
            "x_normal": x,
            "x_skew": rng_gen.exponential(3.0, n),
            "x_corr": 0.995 * x + rng_gen.normal(0.0, 0.02, n),
            "const": np.full(n, 1.0),
            "text": rng_gen.choice(["a", "b"], n),
            "nulls": pl.Series(nulls, dtype=pl.Float64),
            "target": 0.5 * x + rng_gen.normal(0, 0.5, n),
        }
    )
    return frame


def test_basic_counts_and_quantiles(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    assert result.rows == 400
    assert result.columns == len(audit_frame_fixture.columns)
    by_name = {c.name: c for c in result.per_column}
    assert by_name["x_normal"].mean is not None
    assert by_name["x_normal"].q50 is not None
    assert by_name["x_normal"].q01 is not None
    assert math.isclose(by_name["nulls"].null_rate, 0.4, abs_tol=0.05)
    assert by_name["x_normal"].q01 <= by_name["x_normal"].q50 <= by_name["x_normal"].q99


def test_constant_and_id_detection(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    by_name = {c.name: c for c in result.per_column}
    assert by_name["const"].is_constant
    assert by_name["id"].is_id_like
    assert not by_name["x_normal"].is_id_like  # continuous floats never flagged


def test_correlation_pairs_found(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    pair_names = {(p["left"], p["right"]) for p in result.high_corr_pairs}
    assert any({"x_normal", "x_corr"} & set(pair) for pair in pair_names)


def test_outlier_counts_populated(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    by_name = {c.name: c for c in result.per_column}
    assert by_name["x_skew"].iqr_outliers is not None
    assert by_name["x_skew"].zscore_outliers is not None
    assert by_name["x_skew"].iqr_outliers >= 0


def test_target_leakage_flagged_by_correlation(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    # Make a column a copy of the target → identity + correlation flags.
    frame = audit_frame_fixture.with_columns(pl.col("target").alias("target_copy"))
    result = audit_frame(frame, cfg)
    flagged = {f["column"] for f in result.leakage_flags}
    assert "target_copy" in flagged


def test_duplicate_count_exact(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    duplicated = pl.concat([audit_frame_fixture, audit_frame_fixture.head(10)])
    result = audit_frame(duplicated, cfg)
    assert result.duplicate_scan_mode == "exact"
    assert result.duplicate_row_count == 20  # 10 originals + 10 copies


def test_duplicate_count_sampled_when_large(rng_gen: np.random.Generator) -> None:
    n = 5000
    frame = pl.DataFrame({"a": rng_gen.normal(0, 1, n), "b": rng_gen.uniform(0, 1, n)})
    cfg = load_config()
    cfg.audit.max_duplicate_scan_rows = 1000
    result = audit_frame(frame, cfg)
    assert result.duplicate_scan_mode == "sampled"
    assert result.duplicate_scan_note is not None


def test_leakage_name_pattern_flag() -> None:
    frame = pl.DataFrame({"target_secret": [1.0, 2.0], "v": [3.0, 4.0]})
    cfg = load_config()
    result = audit_frame(frame, cfg)
    flagged = {f["column"] for f in result.leakage_flags}
    assert "target_secret" in flagged


def test_json_output_is_strictly_valid(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    # json.dumps with allow_nan=False raises on NaN/inf → guarantees strict JSON.
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert "columns" in payload


def test_markdown_renders_tables(audit_frame_fixture: pl.DataFrame) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = audit_frame(audit_frame_fixture, cfg)
    md = result.to_markdown()
    assert md.startswith("# Data Audit Report")
    assert "| x_normal |" in md
    assert "Highly correlated pairs" in md
    assert "never changes pipeline behavior" in md


def test_entirely_null_column_warns() -> None:
    frame = pl.DataFrame({"void": pl.Series([None, None, None], dtype=pl.Float64)})
    cfg = load_config()
    result = audit_frame(frame, cfg)
    assert any("entirely null" in w for w in result.warnings)


def test_corr_analysis_skipped_when_too_wide(rng_gen: np.random.Generator) -> None:
    n = 50
    data = {f"c{i}": rng_gen.normal(0, 1, n) for i in range(12)}
    frame = pl.DataFrame(data)
    cfg = load_config()
    cfg.audit.max_corr_columns = 10
    result = audit_frame(frame, cfg)
    assert any("correlation analysis skipped" in w for w in result.warnings)
