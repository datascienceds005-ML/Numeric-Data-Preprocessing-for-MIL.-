"""Train/val/test splitting: random, stratified, group-aware, chronological.

Design rules enforced here (DESIGN §4):

* Splitting happens **before any fitted transformation**. The outputs are the
  *only* frames that ever carry a partition tag, and only :func:`split_data`
  can produce tagged frames.
* Sizes are computed with the **largest-remainder method** so they sum exactly
  to ``n`` and every partition is non-empty (guaranteed for ``n >= 3``; smaller
  inputs fail with guidance).
* Stratified splitting falls back to random (with a recorded warning) when any
  class has fewer members than there are partitions, or when sklearn's
  stratifier rejects the data — rare classes must never crash a run.
* Group splitting guarantees zero group overlap; a deterministic retry loop
  (``seed + attempt``) handles degenerate group layouts before falling back.
* Chronological splitting is a contiguous, never-shuffled slice of the
  time-sorted frame (``__row_index__`` is the deterministic tie-break).
* Post-conditions are hard-asserted: zero row-id overlap, zero group-key
  overlap (group strategy), plus a near-duplicate collision report across
  partitions (report-only, see DESIGN §8.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from numprep.config import NumPrepConfig
from numprep.errors import InputContractError, PipelineError
from numprep.ingest import ROW_INDEX_COL
from numprep.partitions import Partition, tag_partition
from numprep.seed import rng

_NEAR_DUP_DECIMALS = 6
_NEAR_DUP_MAX_ROWS = 1_000_000
_GROUP_RETRY_ATTEMPTS = 100


@dataclass(frozen=True)
class SplitMeta:
    """Everything the report and manifest need to know about the split."""

    strategy: str
    requested_ratios: dict[str, float]
    applied_ratios: dict[str, float]
    ratios_normalized: bool
    sizes: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    fallback: str | None = None
    target_task: str | None = None
    target_distribution: dict[str, dict[str, Any]] = field(default_factory=dict)
    group_overlap_zero: bool | None = None
    time_ranges: dict[str, dict[str, str | None]] | None = None
    near_duplicate_collisions: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SplitResult:
    """Tagged partitions plus split metadata.

    ``train`` is the only frame accepted by any ``fit`` call in the pipeline
    (enforced by :mod:`numprep.partitions`).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    meta: SplitMeta

    def partitions(self) -> dict[str, pd.DataFrame]:
        """Mapping of partition name → frame (report/manifest friendly)."""
        return {"train": self.train, "val": self.val, "test": self.test}


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _largest_remainder_sizes(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate integer sizes summing to exactly ``n``, each at least 1."""
    raw = {key: n * value for key, value in ratios.items()}
    floors = {key: int(np.floor(value)) for key, value in raw.items()}
    remainder = n - sum(floors.values())
    # Distribute the remainder by descending fractional part (deterministic
    # tie-break: fixed key order train, val, test).
    order = sorted(ratios, key=lambda k: (-(raw[k] - floors[k]), list(ratios).index(k)))
    sizes = dict(floors)
    for key in order[:remainder]:
        sizes[key] += 1
    # Guarantee non-emptiness when n >= 3 by stealing from the largest.
    for key, size in sizes.items():
        if size < 1:
            donor = max(sizes, key=lambda k: sizes[k])
            if sizes[donor] <= 1:
                raise PipelineError(f"cannot allocate a non-empty {key!r} partition from {n} rows")
            sizes[donor] -= 1
            sizes[key] = 1
    return sizes


# ---------------------------------------------------------------------------
# Task detection & reporting helpers
# ---------------------------------------------------------------------------


def detect_task(y: pd.Series) -> str:
    """Classify a target as ``classification`` or ``regression`` (heuristic).

    Non-float dtypes (int/bool/object/category) are classification; floats are
    classification only when every value is integral and there are ≤ 20 uniques.
    """
    if pd.api.types.is_float_dtype(y):
        non_null = y.dropna()
        if non_null.empty:
            return "regression"
        integral = bool(np.allclose(non_null, np.round(non_null), equal_nan=False))
        if integral and y.nunique() <= 20:
            return "classification"
        return "regression"
    return "classification"


def _target_distribution(y: pd.Series, task: str) -> dict[str, Any]:
    """Per-partition target summary (class proportions or continuous stats)."""
    non_null = y.dropna()
    if task == "classification":
        counts = non_null.astype(str).value_counts().sort_index()
        total = int(counts.sum())
        return {
            "class_proportions": {
                str(value): round(float(count) / total, 6) if total else 0.0
                for value, count in counts.items()
            }
        }
    if non_null.empty:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(non_null.mean()),
        "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
        "min": float(non_null.min()),
        "max": float(non_null.max()),
    }


def _near_duplicate_collisions(
    df: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
) -> dict[str, int]:
    """Count rounded-value hash collisions between partitions (report-only)."""
    subset = [c for c in df.columns if c != ROW_INDEX_COL]
    if not subset or len(df) > _NEAR_DUP_MAX_ROWS:
        return {}
    hashes: dict[str, set[np.uint64]] = {}
    for name, part in parts.items():
        rounded = part[subset].round(_NEAR_DUP_DECIMALS)
        hashes[name] = set(pd.util.hash_pandas_object(rounded, index=False).to_numpy())
    collisions: dict[str, int] = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        shared = hashes[left] & hashes[right]
        if shared:
            collisions[f"{left}|{right}"] = len(shared)
    return collisions


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _random_indices(n: int, sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Shuffled, disjoint index slices in partition order."""
    permutation = rng(seed).permutation(n)
    out: dict[str, np.ndarray] = {}
    cursor = 0
    for key in ("train", "val", "test"):
        size = sizes[key]
        out[key] = np.sort(permutation[cursor : cursor + size])
        cursor += size
    return out


def _stratified_indices(
    y: pd.Series, sizes: dict[str, int], seed: int
) -> tuple[dict[str, np.ndarray], str | None]:
    """Two-stage stratified split; returns indices or a fallback reason."""
    classes = y.dropna().astype(str)
    min_class_count = int(classes.value_counts().min()) if not classes.empty else 0
    if min_class_count < 3:
        return {}, f"rare class with {min_class_count} member(s) cannot be stratified"
    try:
        all_idx = np.arange(len(y))
        train_idx, rest_idx = train_test_split(
            all_idx,
            train_size=sizes["train"],
            stratify=y.astype(str),
            random_state=seed,
            shuffle=True,
        )
        rest_ratio_val = sizes["val"] / max(sizes["val"] + sizes["test"], 1)
        val_idx, test_idx = train_test_split(
            rest_idx,
            train_size=rest_ratio_val,
            stratify=y.iloc[rest_idx].astype(str),
            random_state=seed + 1,
            shuffle=True,
        )
    except ValueError as exc:
        return {}, f"sklearn stratification rejected the data: {exc}"
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }, None


def _group_indices(
    group_keys: pd.Series, fractions: dict[str, float], seed: int
) -> tuple[dict[str, np.ndarray], str | None]:
    """Two-stage group-aware split with deterministic retries; never mixes groups."""
    unique_groups = group_keys.dropna().unique()
    if len(unique_groups) < 3:
        return {}, (f"only {len(unique_groups)} group(s) present; need at least one per partition")
    for attempt in range(_GROUP_RETRY_ATTEMPTS):
        split_state = seed + attempt
        gss1 = GroupShuffleSplit(
            n_splits=1, train_size=fractions["train"], random_state=split_state
        )
        train_idx, rest_idx = next(gss1.split(group_keys, groups=group_keys))
        rest_ratio_val = fractions["val"] / max(fractions["val"] + fractions["test"], 1e-12)
        gss2 = GroupShuffleSplit(
            n_splits=1, train_size=rest_ratio_val, random_state=split_state + 10_000
        )
        val_idx, test_idx = next(
            gss2.split(group_keys.iloc[rest_idx], groups=group_keys.iloc[rest_idx])
        )
        val_idx = rest_idx[val_idx]
        test_idx = rest_idx[test_idx]
        train_groups = set(group_keys.iloc[train_idx])
        val_groups = set(group_keys.iloc[val_idx])
        test_groups = set(group_keys.iloc[test_idx])
        if not val_idx.size or not test_idx.size:
            continue
        if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
            continue  # GroupShuffleSplit guarantees this, the check is belt-and-braces
        return {
            "train": np.sort(train_idx),
            "val": np.sort(val_idx),
            "test": np.sort(test_idx),
        }, None
    return {}, (
        f"group splitting failed to produce disjoint non-empty partitions in "
        f"{_GROUP_RETRY_ATTEMPTS} deterministic attempts"
    )


def _chronological_order(df: pd.DataFrame, cfg: NumPrepConfig) -> tuple[pd.DataFrame, int]:
    """Parse the time column, drop null times, and stably sort by time."""
    time_col = cfg.split.time_column
    assert time_col is not None
    frame = df.copy(deep=False)
    if time_col not in frame.columns:
        raise InputContractError(f"split.time_column {time_col!r} is absent from the data")
    if not pd.api.types.is_datetime64_any_dtype(frame[time_col]):
        frame[time_col] = pd.to_datetime(
            frame[time_col], format=cfg.split.time_format, errors="coerce", utc=True
        )
    dropped = int(frame[time_col].isna().sum())
    if dropped:
        frame = frame.dropna(subset=[time_col])
    frame = frame.sort_values([time_col, ROW_INDEX_COL], kind="mergesort")
    return frame, dropped


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def split_data(df: pd.DataFrame, cfg: NumPrepConfig) -> SplitResult:
    """Split the cleaned frame into tagged train/val/test partitions.

    Raises :class:`PipelineError` for degenerate inputs (n < 3) and
    :class:`InputContractError` when configured key columns are absent.
    """
    n = len(df)
    if n < 3:
        raise PipelineError(
            f"cannot form three non-empty partitions from {n} rows; "
            "provide at least 3 rows (larger datasets strongly recommended)"
        )
    if ROW_INDEX_COL not in df.columns:
        raise PipelineError(
            f"internal invariant violated: {ROW_INDEX_COL!r} column is missing before split"
        )

    requested = cfg.split.ratios
    total = sum(requested.values())
    applied = {key: value / total for key, value in requested.items()}
    ratios_normalized = any(abs(applied[k] - requested[k]) > 1e-12 for k in requested)
    warnings: list[str] = []
    if ratios_normalized:
        warnings.append(
            f"split.ratios {requested} did not sum to 1 and were renormalized to "
            f"{ {k: round(v, 6) for k, v in applied.items()} }"
        )

    sizes = _largest_remainder_sizes(n, applied)
    strategy: str = cfg.split.strategy
    fallback: str | None = None

    if strategy == "random":
        indices = _random_indices(n, sizes, cfg.run.seed)
    elif strategy == "stratified":
        target = cfg.target.column
        assert target is not None  # config-validated
        if target not in df.columns:
            raise InputContractError(f"target column {target!r} is absent from the data")
        indices, fallback = _stratified_indices(df[target], sizes, cfg.run.seed)
        if fallback:
            warnings.append(f"stratified split fell back to random: {fallback}")
            indices = _random_indices(n, sizes, cfg.run.seed)
            strategy = "random (stratified fallback)"
    elif strategy == "group":
        group_cols = cfg.split.group_columns
        absent = [c for c in group_cols if c not in df.columns]
        if absent:
            raise InputContractError(f"split.group_columns references absent columns: {absent}")
        keys = df[group_cols].astype(str)
        group_keys = keys.iloc[:, 0]
        for col in keys.columns[1:]:
            group_keys = group_keys + "|" + keys[col]
        indices, fallback = _group_indices(group_keys, applied, cfg.run.seed)
        if fallback:
            warnings.append(f"group split fell back to random: {fallback}")
            indices = _random_indices(n, sizes, cfg.run.seed)
            strategy = "random (group fallback)"
    elif strategy == "chronological":
        ordered, dropped_times = _chronological_order(df, cfg)
        if dropped_times:
            warnings.append(
                f"{dropped_times} row(s) with null/unparseable {cfg.split.time_column!r} "
                "were dropped before the chronological split"
            )
        n_ordered = len(ordered)
        if n_ordered < 3:
            raise PipelineError(
                f"only {n_ordered} rows have a parseable timestamp; "
                "cannot form three chronological partitions"
            )
        sizes = _largest_remainder_sizes(n_ordered, applied)
        df = ordered.reset_index(drop=True)
        # Contiguous positional slices of the sorted frame: chronological splits
        # are never shuffled and never interleaved.
        indices = {}
        cursor = 0
        for key in ("train", "val", "test"):
            size = sizes[key]
            indices[key] = np.arange(cursor, cursor + size)
            cursor += size
    else:  # pragma: no cover - guarded by config validation
        raise PipelineError(f"unknown split strategy: {strategy}")

    parts: dict[str, pd.DataFrame] = {}
    for key, role in (("train", Partition.TRAIN), ("val", Partition.VAL), ("test", Partition.TEST)):
        part = df.iloc[indices[key]].reset_index(drop=True)
        parts[key] = tag_partition(part, role)

    # Hard post-conditions -------------------------------------------------
    row_ids = {key: set(parts[key][ROW_INDEX_COL].tolist()) for key in parts}
    if (
        row_ids["train"] & row_ids["val"]
        or row_ids["train"] & row_ids["test"]
        or row_ids["val"] & row_ids["test"]
    ):
        raise PipelineError("split invariant violated: row overlap across partitions")

    group_overlap_zero: bool | None = None
    if cfg.split.strategy == "group" and not fallback:
        group_cols = cfg.split.group_columns

        def _keys(frame: pd.DataFrame) -> set[tuple[str, ...]]:
            return set(map(tuple, frame[group_cols].astype(str).to_numpy()))

        train_keys, val_keys, test_keys = (
            _keys(parts["train"]),
            _keys(parts["val"]),
            _keys(parts["test"]),
        )
        group_overlap_zero = not (
            train_keys & val_keys or train_keys & test_keys or val_keys & test_keys
        )
        if not group_overlap_zero:
            raise PipelineError("split invariant violated: group overlap across partitions")

    meta_task = None
    target_distribution: dict[str, dict[str, Any]] = {}
    if cfg.target.column is not None and cfg.target.column in df.columns:
        y = df[cfg.target.column]
        meta_task = detect_task(y)
        for key in ("train", "val", "test"):
            target_distribution[key] = _target_distribution(
                parts[key][cfg.target.column], meta_task
            )

    time_ranges = None
    if cfg.split.strategy == "chronological":
        time_col = cfg.split.time_column
        assert time_col is not None
        time_ranges = {
            key: {
                "min": str(parts[key][time_col].min()) if len(parts[key]) else None,
                "max": str(parts[key][time_col].max()) if len(parts[key]) else None,
            }
            for key in ("train", "val", "test")
        }

    near_dups = _near_duplicate_collisions(df, parts)

    meta = SplitMeta(
        strategy=strategy,
        requested_ratios=dict(requested),
        applied_ratios={k: round(v, 6) for k, v in applied.items()},
        ratios_normalized=ratios_normalized,
        sizes={k: len(parts[k]) for k in ("train", "val", "test")},
        warnings=warnings,
        fallback=fallback,
        target_task=meta_task,
        target_distribution=target_distribution,
        group_overlap_zero=group_overlap_zero,
        time_ranges=time_ranges,
        near_duplicate_collisions=near_dups,
    )
    return SplitResult(train=parts["train"], val=parts["val"], test=parts["test"], meta=meta)
