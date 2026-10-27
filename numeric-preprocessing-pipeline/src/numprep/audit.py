"""Audit / profiling: fast vectorized data-quality report (JSON + Markdown).

The audit is **report-only**: nothing in the pipeline consumes its output to make
fitted decisions (see DESIGN §4.2 — this separation is asserted by a leakage test).
All aggregations run in one or two vectorized Polars passes; correlation and
duplicate analyses fall back to deterministic strided sampling above configured
row/column caps, and every approximation is labeled in the report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from numprep.config import NumPrepConfig
from numprep.ingest import ROW_INDEX_COL
from numprep.sampling import strided_sample_indices

_QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


def _to_float(value: object) -> float:
    """Narrow a pandas/numpy scalar to float (raises on genuinely non-numeric)."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"expected a numeric scalar, got {type(value).__name__}")


def _finite_or_none(value: float | None) -> float | None:
    """Convert NaN/inf to ``None`` so the JSON stays strictly valid."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


@dataclass(frozen=True)
class ColumnAudit:
    """Per-column audit statistics."""

    name: str
    dtype: str
    role: str  # numeric | boolean | string | index
    null_count: int
    null_rate: float
    n_unique: int | None
    unique_ratio: float | None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q01: float | None = None
    q25: float | None = None
    q50: float | None = None
    q75: float | None = None
    q99: float | None = None
    max: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    zeros: int | None = None
    negatives: int | None = None
    inf_count: int | None = None
    nan_count: int | None = None
    iqr_low_fence: float | None = None
    iqr_high_fence: float | None = None
    iqr_outliers: int | None = None
    zscore_outliers: int | None = None
    is_constant: bool = False
    is_near_constant: bool = False
    is_id_like: bool = False
    id_reason: str | None = None
    leakage_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AuditResult:
    """Complete audit payload (machine-readable, rendered to MD by ``to_markdown``)."""

    rows: int
    columns: int
    memory_mb: float
    per_column: list[ColumnAudit]
    duplicate_row_count: int | None
    duplicate_scan_mode: str  # exact | sampled | skipped
    duplicate_scan_note: str | None
    high_corr_pairs: list[dict[str, Any]]
    leakage_flags: list[dict[str, Any]]
    warnings: list[str]
    target_column: str | None
    quantiles_used: tuple[float, ...] = _QUANTILES

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation (NaN/inf removed, quantiles as list)."""
        return {
            "rows": self.rows,
            "columns": self.columns,
            "memory_mb_estimate": round(self.memory_mb, 2),
            "target_column": self.target_column,
            "duplicate_rows": {
                "count": self.duplicate_row_count,
                "mode": self.duplicate_scan_mode,
                "note": self.duplicate_scan_note,
            },
            "high_correlation_pairs": self.high_corr_pairs,
            "potential_target_leakage": self.leakage_flags,
            "warnings": self.warnings,
            "quantiles_used": list(self.quantiles_used),
            "column_stats": [c.to_dict() for c in self.per_column],
        }

    def to_markdown(self) -> str:
        """Human-readable Markdown rendering of the audit."""
        lines: list[str] = []
        lines.append("# Data Audit Report")
        lines.append("")
        lines.append(f"- **Rows:** {self.rows:,}")
        lines.append(f"- **Columns:** {self.columns}")
        lines.append(f"- **Estimated in-memory size:** {self.memory_mb:,.1f} MiB")
        lines.append(f"- **Target column:** {self.target_column or '(none)'}")
        if self.duplicate_row_count is not None:
            label = "exact" if self.duplicate_scan_mode == "exact" else "sampled estimate"
            lines.append(f"- **Duplicate rows ({label}):** {self.duplicate_row_count:,}")
        if self.duplicate_scan_note:
            lines.append(f"- **Duplicate scan note:** {self.duplicate_scan_note}")
        lines.append("")

        lines.append("## Column profile")
        lines.append("")
        lines.append(
            "| column | dtype | null % | unique | mean | std | min | q01 | q50 | q99 | max "
            "| skew | IQR outliers | z>3 | flags |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for c in self.per_column:
            flags = []
            if c.is_constant:
                flags.append("constant")
            if c.is_near_constant:
                flags.append("near-constant")
            if c.is_id_like:
                flags.append("id-like")
            if c.leakage_flags:
                flags.append("leak?" + ";".join(c.leakage_flags))
            lines.append(
                f"| {c.name} | {c.dtype} | {c.null_rate * 100:.2f} | "
                f"{c.n_unique if c.n_unique is not None else '—'} | "
                f"{_fmt(c.mean)} | {_fmt(c.std)} | {_fmt(c.min)} | {_fmt(c.q01)} | "
                f"{_fmt(c.q50)} | {_fmt(c.q99)} | {_fmt(c.max)} | {_fmt(c.skew)} | "
                f"{c.iqr_outliers if c.iqr_outliers is not None else '—'} | "
                f"{c.zscore_outliers if c.zscore_outliers is not None else '—'} | "
                f"{','.join(flags) if flags else '—'} |"
            )
        lines.append("")

        if self.high_corr_pairs:
            lines.append("## Highly correlated pairs")
            lines.append("")
            lines.append("| left | right | pearson r |")
            lines.append("|---|---|---:|")
            for pair in self.high_corr_pairs:
                lines.append(f"| {pair['left']} | {pair['right']} | {pair['r']:.4f} |")
            lines.append("")
        else:
            lines.append("## Highly correlated pairs")
            lines.append("")
            lines.append(
                "_No pairs above the configured threshold (or analysis skipped; see warnings)._"
            )
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        lines.append(
            "_The audit is informational: it never changes pipeline behavior. All "
            "fitted decisions are made on the train partition only._"
        )
        lines.append("")
        return "\n".join(lines)


def _fmt(value: float | int | None) -> str:
    """Compact float formatting for Markdown tables."""
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if math.isnan(value) or math.isinf(value):
        return "—"
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _estimate_memory_mb(df: pl.DataFrame) -> float:
    """Rough in-memory estimate: exact for numeric/bool, sampled for strings."""
    if df.height == 0:
        return 0.0
    total_bytes = 0.0
    for name, dtype in df.schema.items():
        if dtype == pl.String:
            sample_idx = strided_sample_indices(df.height, 10_000)
            sample = df[name].gather(pl.Series(sample_idx)).drop_nulls()
            raw_mean = sample.str.len_bytes().mean() if sample.len() else None
            avg_len = _to_float(raw_mean) if raw_mean is not None else 0.0
            if math.isnan(avg_len):
                avg_len = 0.0
            total_bytes += (avg_len + 4) * df.height  # polars view: data + offsets
        else:
            total_bytes += _dtype_size_bytes(dtype) * df.height
    return total_bytes / (1024.0 * 1024.0)


def _dtype_size_bytes(dtype: pl.DataType) -> int:
    """Byte width of a non-string dtype (sampled-safe fallback for unknown types)."""
    for attr in ("base_size", "item_size"):
        value = getattr(dtype, attr, None)
        if value is not None:
            return int(value)
    return 8


def _first_pass(df: pl.DataFrame, numeric: list[str], other: list[str]) -> pl.DataFrame:
    """One vectorized select with every per-column aggregate."""
    exprs: list[pl.Expr] = []
    for name in numeric:
        col = pl.col(name)
        exprs.extend(
            [
                col.count().alias(f"{name}|nonnull"),
                col.null_count().alias(f"{name}|nulls"),
                col.n_unique().alias(f"{name}|nuniq"),
                col.min().alias(f"{name}|min"),
                col.max().alias(f"{name}|max"),
                col.mean().alias(f"{name}|mean"),
                col.std().alias(f"{name}|std"),
                col.skew().alias(f"{name}|skew"),
                col.kurtosis().alias(f"{name}|kurt"),
                (col == 0).sum().alias(f"{name}|zeros"),
                (col < 0).sum().alias(f"{name}|neg"),
                col.is_infinite().sum().alias(f"{name}|inf"),
                col.is_nan().sum().alias(f"{name}|nan"),
            ]
        )
        for q in _QUANTILES:
            exprs.append(col.quantile(q, interpolation="linear").alias(f"{name}|q{q}"))
    for name in other:
        col = pl.col(name)
        exprs.extend(
            [
                col.count().alias(f"{name}|nonnull"),
                col.null_count().alias(f"{name}|nulls"),
                col.n_unique().alias(f"{name}|nuniq"),
            ]
        )
    return df.select(exprs)


def _column_roles(df: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Partition columns into (numeric, boolean, other) — robust to exotic dtypes."""
    numeric: list[str] = []
    boolean: list[str] = []
    other: list[str] = []
    for name, dtype in df.schema.items():
        if name == ROW_INDEX_COL:
            continue
        if dtype.is_numeric():
            numeric.append(name)
        elif dtype == pl.Boolean:
            boolean.append(name)
        else:
            other.append(name)  # String, Object, Categorical, Datetime, ...
    return numeric, boolean, other


def _outlier_pass(df: pl.DataFrame, numeric: list[str]) -> pl.DataFrame:
    """Second pass: IQR-fence and z-score outlier counts using literal fences."""
    exprs: list[pl.Expr] = []
    for name in numeric:
        col = pl.col(name)
        q25 = pl.col(name).quantile(0.25, interpolation="linear")
        q75 = pl.col(name).quantile(0.75, interpolation="linear")
        iqr = q75 - q25
        mean = pl.col(name).mean()
        std = pl.col(name).std()
        exprs.extend(
            [
                (q25 - 1.5 * iqr).alias(f"{name}|iqr_low"),
                (q75 + 1.5 * iqr).alias(f"{name}|iqr_high"),
                ((col < (q25 - 1.5 * iqr)) | (col > (q75 + 1.5 * iqr)))
                .fill_null(False)
                .sum()
                .alias(f"{name}|iqr_out"),
                ((col - mean).abs() > 3.0 * std).fill_null(False).sum().alias(f"{name}|z_out"),
            ]
        )
    return df.select(exprs)


def _count_duplicate_rows(
    df: pl.DataFrame, cfg: NumPrepConfig
) -> tuple[int | None, str, str | None]:
    """Exact duplicate-row count, or a labeled strided-sampled estimate."""
    subset = [c for c in df.columns if c != ROW_INDEX_COL]
    if not subset:
        return 0, "exact", None
    if df.height > cfg.audit.max_duplicate_scan_rows:
        idx = strided_sample_indices(df.height, 200_000)
        sample = df.gather(pl.Series(idx))
        grouped = sample.group_by(subset).len()
        dup = int(grouped.filter(pl.col("len") > 1).select(pl.col("len").sum().fill_null(0)).item())
        note = (
            f"computed on a deterministic strided sample of {sample.height:,} rows "
            f"(full frame has {df.height:,}; audit.max_duplicate_scan_rows="
            f"{cfg.audit.max_duplicate_scan_rows:,})"
        )
        return dup, "sampled", note
    grouped = df.group_by(subset).len()
    dup = int(grouped.filter(pl.col("len") > 1).select(pl.col("len").sum().fill_null(0)).item())
    return dup, "exact", None


def _correlation_analysis(
    df: pl.DataFrame,
    numeric: list[str],
    cfg: NumPrepConfig,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Pearson correlations on a strided sample; returns pairs and per-column r vs target."""
    pairs: list[dict[str, Any]] = []
    target_corr: dict[str, float] = {}
    target = cfg.target.column

    if len(numeric) > cfg.audit.max_corr_columns:
        warnings.append(
            f"{len(numeric)} numeric columns exceed audit.max_corr_columns="
            f"{cfg.audit.max_corr_columns}; correlation analysis skipped"
        )
        return pairs, target_corr

    idx = strided_sample_indices(df.height, cfg.audit.sample_size)
    sampled = df.gather(pl.Series(idx)) if df.height else df
    if sampled.height < 2:
        return pairs, target_corr

    analysis_cols = [c for c in numeric if c != ROW_INDEX_COL]
    frame: pd.DataFrame = sampled.select(analysis_cols).to_pandas()
    corr = frame.corr(method="pearson")

    for i, left in enumerate(analysis_cols):
        for right in analysis_cols[i + 1 :]:
            r = corr.loc[left, right]
            if pd.isna(r):
                continue
            r_value = _to_float(r)
            if abs(r_value) >= cfg.audit.corr_threshold:
                pairs.append({"left": left, "right": right, "r": r_value})

    if target is not None and target in analysis_cols:
        for col in analysis_cols:
            if col == target:
                continue
            r = corr.loc[col, target]
            if not pd.isna(r):
                target_corr[col] = _to_float(r)

    pairs.sort(key=lambda p: (-abs(p["r"]), p["left"], p["right"]))
    return pairs, target_corr


def audit_frame(df: pl.DataFrame, cfg: NumPrepConfig) -> AuditResult:
    """Profile the frame and return the structured audit result."""
    warnings: list[str] = []
    numeric, boolean, other = _column_roles(df)
    stats = _first_pass(df, numeric, other).row(0, named=True)

    id_name_re = re.compile(cfg.cleaning.id_name_pattern)
    leak_name_re = re.compile(cfg.audit.leakage_name_pattern)

    pairs, target_corr = _correlation_analysis(df, numeric, cfg, warnings)
    dup_count, dup_mode, dup_note = _count_duplicate_rows(df, cfg)

    outlier_stats: dict[str, Any] = {}
    if numeric:
        outlier_row = _outlier_pass(df, numeric).row(0, named=True)
        outlier_stats = outlier_row

    target = cfg.target.column
    target_series = df[target] if target is not None and target in df.columns else None

    per_column: list[ColumnAudit] = []
    for name in df.columns:
        if name == ROW_INDEX_COL:
            per_column.append(
                ColumnAudit(
                    name=name,
                    dtype=str(df.schema[name]),
                    role="index",
                    null_count=0,
                    null_rate=0.0,
                    n_unique=int(df.height),
                    unique_ratio=1.0,
                )
            )
            continue
        dtype = df.schema[name]
        role = "numeric" if name in numeric else ("boolean" if name in boolean else "other")
        nulls = int(stats[f"{name}|nulls"])
        nonnull = int(stats[f"{name}|nonnull"])
        nuniq = stats.get(f"{name}|nuniq")
        nuniq_int = int(nuniq) if nuniq is not None else None
        unique_ratio = (nuniq_int / df.height) if (nuniq_int is not None and df.height) else None

        is_constant = nuniq_int is not None and nuniq_int <= 1
        is_near_constant = bool(
            nuniq_int is not None
            and 2 <= nuniq_int <= 10
            and df.height
            and (nuniq_int / df.height) < 0.01
        )

        id_reason: str | None = None
        integer_or_string = dtype.is_integer() or dtype == pl.String
        if id_name_re.search(name):
            id_reason = f"name matches id pattern {cfg.cleaning.id_name_pattern!r}"
        elif (
            integer_or_string
            and nuniq_int is not None
            and nuniq_int > 10
            and unique_ratio is not None
            and unique_ratio >= cfg.cleaning.id_unique_ratio
        ):
            # Only integer/string columns qualify by uniqueness: continuous float
            # columns are legitimately ~100% unique and would all be false positives.
            id_reason = (
                f"{dtype}-typed column with unique ratio {unique_ratio:.4f} "
                f">= {cfg.cleaning.id_unique_ratio}"
            )

        leakage: list[str] = []
        if leak_name_re.search(name) and name != target:
            leakage.append("name matches leakage pattern")
        if name in target_corr and abs(target_corr[name]) >= cfg.audit.leakage_corr_threshold:
            leakage.append(f"|r(target,{name})|={abs(target_corr[name]):.4f}")
        if target_series is not None and name != target and name in numeric and role == "numeric":
            same_mask = (df[name] == target_series) | (df[name].is_null() & target_series.is_null())
            # fill_null: an all-null column must NOT count as "identical" — polars
            # boolean .all() skips nulls, which would otherwise return True.
            if nonnull > 0 and bool(same_mask.fill_null(False).all()):
                leakage.append("column identical to target")

        entry = ColumnAudit(
            name=name,
            dtype=str(dtype),
            role=role,
            null_count=nulls,
            null_rate=nulls / df.height if df.height else 0.0,
            n_unique=nuniq_int,
            unique_ratio=unique_ratio,
            mean=_finite_or_none(stats.get(f"{name}|mean")),
            std=_finite_or_none(stats.get(f"{name}|std")),
            min=_finite_or_none(stats.get(f"{name}|min")),
            q01=_finite_or_none(stats.get(f"{name}|q0.01")),
            q25=_finite_or_none(stats.get(f"{name}|q0.25")),
            q50=_finite_or_none(stats.get(f"{name}|q0.5")),
            q75=_finite_or_none(stats.get(f"{name}|q0.75")),
            q99=_finite_or_none(stats.get(f"{name}|q0.99")),
            max=_finite_or_none(stats.get(f"{name}|max")),
            skew=_finite_or_none(stats.get(f"{name}|skew")),
            kurtosis=_finite_or_none(stats.get(f"{name}|kurt")),
            zeros=int(stats[f"{name}|zeros"]) if f"{name}|zeros" in stats else None,
            negatives=int(stats[f"{name}|neg"]) if f"{name}|neg" in stats else None,
            inf_count=int(stats[f"{name}|inf"]) if f"{name}|inf" in stats else None,
            nan_count=int(stats[f"{name}|nan"]) if f"{name}|nan" in stats else None,
            iqr_low_fence=_finite_or_none(outlier_stats.get(f"{name}|iqr_low")),
            iqr_high_fence=_finite_or_none(outlier_stats.get(f"{name}|iqr_high")),
            iqr_outliers=int(outlier_stats[f"{name}|iqr_out"])
            if f"{name}|iqr_out" in outlier_stats
            else None,
            zscore_outliers=int(outlier_stats[f"{name}|z_out"])
            if f"{name}|z_out" in outlier_stats
            else None,
            is_constant=is_constant,
            is_near_constant=is_near_constant,
            is_id_like=id_reason is not None,
            id_reason=id_reason,
            leakage_flags=leakage,
        )
        per_column.append(entry)

    leakage_flags = [
        {"column": c.name, "reasons": c.leakage_flags} for c in per_column if c.leakage_flags
    ]

    all_null_columns = [c.name for c in per_column if df.height > 0 and c.null_count == df.height]
    for name in all_null_columns:
        warnings.append(f"column {name!r} is entirely null")

    return AuditResult(
        rows=int(df.height),
        columns=len(df.columns),
        memory_mb=_estimate_memory_mb(df),
        per_column=per_column,
        duplicate_row_count=dup_count,
        duplicate_scan_mode=dup_mode,
        duplicate_scan_note=dup_note,
        high_corr_pairs=pairs,
        leakage_flags=leakage_flags,
        warnings=warnings,
        target_column=target,
    )
