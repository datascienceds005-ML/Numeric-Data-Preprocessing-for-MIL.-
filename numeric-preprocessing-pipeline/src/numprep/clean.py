"""Stateless cleaning — operations that are safe to run *before* the split.

Only row-level, parameter-free operations live here:

1. ``±inf → null`` (so extreme values are imputed, never propagated),
2. domain rules (``non_negative_columns``: negatives become null),
3. exact deduplication (``keep="first"``, stable order; runs **after**
   normalisation so rows differing only in inf/NaN sentinels collapse too),
4. past-only temporal features (delegated to :mod:`numprep.temporal` — stateless,
   see DESIGN §4.3 for the leakage reasoning),
5. rows with a null target are dropped (configurable; the only supervised
   row filter, applied before splitting so no partition depends on it).

Every count is recorded and surfaced in the run report. No fitted statistic is
computed here — that is a hard design rule, tested in ``tests/test_leakage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from numprep.config import NumPrepConfig
from numprep.errors import PipelineError
from numprep.ingest import ROW_INDEX_COL
from numprep.temporal import TemporalResult, add_temporal_features


@dataclass(frozen=True)
class CleanStats:
    """Counters for every stateless cleaning operation."""

    rows_in: int
    rows_out: int
    inf_replaced: dict[str, int] = field(default_factory=dict)
    negatives_nulled: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    null_target_rows_dropped: int = 0
    temporal: TemporalResult | None = None
    columns_out: list[str] = field(default_factory=list)


def _null_out_infinities(df: pl.DataFrame, enabled: bool) -> tuple[pl.DataFrame, dict[str, int]]:
    """Replace ``+inf``/``-inf`` with null in numeric columns (counts returned)."""
    counts: dict[str, int] = {}
    if not enabled:
        return df, counts
    numeric = [
        name
        for name, dtype in df.schema.items()
        if dtype.is_float()  # integers cannot hold infinities
    ]
    if not numeric:
        return df, counts
    inf_counts = df.select([pl.col(name).is_infinite().sum().alias(name) for name in numeric]).row(
        0, named=True
    )
    counts = {name: int(value) for name, value in inf_counts.items() if int(value) > 0}
    if not counts:
        return df, counts
    exprs = []
    for name in df.columns:
        col = pl.col(name)
        if name in counts:
            exprs.append(pl.when(col.is_infinite()).then(None).otherwise(col).alias(name))
        else:
            exprs.append(col)
    return df.select(exprs), counts


def _apply_non_negative_rules(
    df: pl.DataFrame, columns: list[str]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Null out negative values in configured columns (domain rules)."""
    counts: dict[str, int] = {}
    present = [name for name in columns if name in df.columns]
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise PipelineError(f"cleaning.non_negative_columns references absent columns: {missing}")
    applicable = [name for name in present if df.schema[name].is_numeric()]
    if not applicable:
        return df, counts
    neg_counts = df.select(
        [pl.col(name).lt(0).fill_null(False).sum().alias(name) for name in applicable]
    ).row(0, named=True)
    counts = {name: int(value) for name, value in neg_counts.items() if int(value) > 0}
    if not counts:
        return df, counts
    exprs = []
    for name in df.columns:
        col = pl.col(name)
        if name in counts:
            exprs.append(pl.when(col < 0).then(None).otherwise(col).alias(name))
        else:
            exprs.append(col)
    return df.select(exprs), counts


def clean_stateless(df: pl.DataFrame, cfg: NumPrepConfig) -> tuple[pl.DataFrame, CleanStats]:
    """Apply all stateless cleaning operations in the documented order.

    Returns the cleaned frame and per-operation counters.
    """
    rows_in = int(df.height)
    cleaning = cfg.cleaning

    df, inf_counts = _null_out_infinities(df, cleaning.inf_to_nan)
    df, neg_counts = _apply_non_negative_rules(df, cleaning.non_negative_columns)

    duplicates_removed = 0
    if cleaning.dedupe:
        # The provenance row-index is excluded: it is added at ingest and must
        # not mask true content duplicates (two copies of a row differ only in
        # their position).
        subset = cleaning.dedupe_subset or [name for name in df.columns if name != ROW_INDEX_COL]
        absent = [name for name in subset if name not in df.columns]
        if absent:
            raise PipelineError(f"cleaning.dedupe_subset references absent columns: {absent}")
        rows_before = int(df.height)
        df = df.unique(subset=subset, keep="first", maintain_order=True)
        duplicates_removed = rows_before - int(df.height)

    temporal_result: TemporalResult | None = None
    if cleaning.temporal.enabled:
        df, temporal_result = add_temporal_features(df, cfg)

    null_target_dropped = 0
    if cfg.target.column is not None and cfg.target.drop_null_rows:
        target = cfg.target.column
        if target not in df.columns:
            raise PipelineError(
                f"target column {target!r} not found; set target.column to null "
                "for unsupervised runs"
            )
        null_target_dropped = int(df[target].null_count())
        df = df.filter(pl.col(target).is_not_null())

    stats = CleanStats(
        rows_in=rows_in,
        rows_out=int(df.height),
        inf_replaced=inf_counts,
        negatives_nulled=neg_counts,
        duplicates_removed=duplicates_removed,
        null_target_rows_dropped=null_target_dropped,
        temporal=temporal_result,
        columns_out=list(df.columns),
    )
    return df, stats
