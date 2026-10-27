"""Past-only lag / rolling features for chronological data.

**Why this runs before the split (and is still leak-free)** — see DESIGN §4.3:
these features are *stateless* (no parameters are estimated anywhere) and use
only rows strictly earlier than the current row (``shift(>=1)`` before every
window). Computing them on the full time-ordered frame gives validation/test
rows their legitimate history — the standard construction in time-series ML —
and cannot move future information into the past.

Semantics
---------
* Rows are sorted by ``split.time_column`` with ``__row_index__`` as the
  deterministic tie-break; the sorted order is kept on the returned frame.
* Rows whose timestamp is null/unparseable are dropped (count returned).
* ``lag(k)`` → value at *t−k* within the same entity group (global when no
  groups configured).
* ``roll{stat}{w}`` → statistic over the window ending at *t−1*
  (``shift(1)`` then ``rolling_{stat}(w, min_samples=w)``) — early rows and
  group boundaries produce nulls, which the train-fitted imputer resolves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

from numprep.config import NumPrepConfig
from numprep.errors import InputContractError, PipelineError
from numprep.ingest import ROW_INDEX_COL

_ROLLING = {
    "mean": pl.Expr.rolling_mean,
    "std": pl.Expr.rolling_std,
    "min": pl.Expr.rolling_min,
    "max": pl.Expr.rolling_max,
}


@dataclass(frozen=True)
class TemporalResult:
    """Outcome of the temporal feature stage."""

    added_columns: list[str] = field(default_factory=list)
    dropped_null_time_rows: int = 0
    time_min: str | None = None
    time_max: str | None = None


def _resolve_feature_columns(df: pl.DataFrame, cfg: NumPrepConfig) -> list[str]:
    """Pick lag/rolling source columns: configured list, else all numeric features.

    The auto rule excludes the row index, the time axis, the target, and
    ID-like columns (name pattern) — lagging an ID is noise by construction.
    """
    temporal = cfg.cleaning.temporal
    id_name_re = re.compile(cfg.cleaning.id_name_pattern)
    excluded = {
        ROW_INDEX_COL,
        cfg.split.time_column,
        cfg.target.column,
    }
    if temporal.columns:
        missing = [c for c in temporal.columns if c not in df.columns]
        if missing:
            raise PipelineError(f"cleaning.temporal.columns references absent columns: {missing}")
        return list(temporal.columns)
    return [
        name
        for name, dtype in df.schema.items()
        if dtype.is_numeric() and name not in excluded and not id_name_re.search(name)
    ]


def _ensure_time_column(df: pl.DataFrame, cfg: NumPrepConfig) -> tuple[pl.DataFrame, int]:
    """Parse the time column to a temporal dtype when needed; drop unparseable rows."""
    time_col = cfg.split.time_column
    if time_col not in df.columns:
        raise InputContractError(f"split.time_column {time_col!r} is absent from the data")
    dropped = 0
    dtype = df.schema[time_col]
    if dtype == pl.String:
        parsed = (
            pl.col(time_col)
            .str.to_datetime(format=cfg.split.time_format, strict=False, exact=True)
            .alias(time_col)
        )
        df = df.with_columns(parsed)
    elif not (dtype.is_temporal() or dtype.is_numeric()):
        raise PipelineError(
            f"time column {time_col!r} has unsupported dtype {dtype}; "
            "expected temporal, numeric, or ISO-8601 strings"
        )
    dropped = int(df[time_col].null_count())
    if dropped:
        df = df.filter(pl.col(time_col).is_not_null())
    return df, dropped


def add_temporal_features(
    df: pl.DataFrame, cfg: NumPrepConfig
) -> tuple[pl.DataFrame, TemporalResult]:
    """Add past-only lag/rolling features; returns the frame and a summary."""
    temporal = cfg.cleaning.temporal
    df, dropped = _ensure_time_column(df, cfg)
    time_col = cfg.split.time_column
    assert time_col is not None  # guaranteed by config cross-field validation

    df = df.sort([time_col, ROW_INDEX_COL], maintain_order=True)

    feature_cols = _resolve_feature_columns(df, cfg)
    group_cols = [c for c in temporal.group_columns if c in df.columns]
    missing_groups = [c for c in temporal.group_columns if c not in df.columns]
    if missing_groups:
        raise InputContractError(
            f"cleaning.temporal.group_columns references absent columns: {missing_groups}"
        )

    added: list[str] = []
    new_columns: list[pl.Expr] = []

    for col in feature_cols:
        for lag in temporal.lags:
            expr = pl.col(col).shift(lag)
            if group_cols:
                expr = expr.over(group_cols)
            name = f"{col}__lag{lag}"
            new_columns.append(expr.alias(name))
            added.append(name)
        for window in temporal.rolling_windows:
            for stat in temporal.rolling_stats:
                shifted = pl.col(col).shift(1)
                rolled = _ROLLING[stat](shifted, window_size=window, min_samples=window)
                if group_cols:
                    rolled = rolled.over(group_cols)
                name = f"{col}__roll{stat}{window}"
                new_columns.append(rolled.alias(name))
                added.append(name)

    if new_columns:
        df = df.with_columns(new_columns)

    time_min = time_max = None
    if df.height:
        bounds = df.select(
            pl.col(time_col).min().alias("mn"), pl.col(time_col).max().alias("mx")
        ).row(0, named=True)
        time_min = str(bounds["mn"])
        time_max = str(bounds["mx"])

    return df, TemporalResult(
        added_columns=added,
        dropped_null_time_rows=dropped,
        time_min=time_min,
        time_max=time_max,
    )
