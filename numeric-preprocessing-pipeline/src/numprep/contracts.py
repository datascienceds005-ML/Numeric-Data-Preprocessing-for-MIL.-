"""Input and output data-contract validation.

Two layers, deliberately different:

* **Input contract** — fast *exact* native checks on the full Polars frame
  (required columns, expected dtypes, null-rate ceilings) plus **pandera**
  (pandas backend) validation of element-wise range checks on a deterministic
  strided sample (bounded by ``validation.input.sample_rows``) so huge frames do
  not need a second full pandas copy just for validation.
* **Output contract** — exact, full-frame native checks on the produced
  partitions: schema consistency across splits, non-empty splits, no NaN/inf,
  numeric-only features. Nothing about the outputs is sampled: the contract must
  hold for every row we ship.

All failures aggregate into one actionable error before raising, so a single run
reports every violation at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from numprep.config import InputContractConfig, NumPrepConfig
from numprep.errors import InputContractError, OutputContractError
from numprep.ingest import ROW_INDEX_COL
from numprep.sampling import strided_sample_indices

_MAX_REPORTED_FAILURES = 20


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def _build_pandera_schema(contract: InputContractConfig, columns: list[str]) -> DataFrameSchema:
    """Build a pandera schema covering the declared column ranges.

    Columns without a configured range are deliberately absent from the schema:
    presence/dtype/null-rate rules are enforced exactly by the native layer.
    """
    column_schemas: dict[str, Column] = {}
    for name in columns:
        range_spec = contract.ranges.get(name)
        if range_spec is None:
            continue
        checks: list[Check] = []
        if range_spec.min is not None and range_spec.max is not None:
            checks.append(Check.in_range(range_spec.min, range_spec.max))
        elif range_spec.min is not None:
            checks.append(Check.ge(range_spec.min))
        elif range_spec.max is not None:
            checks.append(Check.le(range_spec.max))
        column_schemas[name] = Column(
            "float64",
            nullable=True,
            checks=checks,
            coerce=True,
        )
    return DataFrameSchema(
        column_schemas,
        strict=False,
        coerce=False,
        ordered=False,
    )


def _sampled_pandas_frame(df: pl.DataFrame, max_rows: int) -> pd.DataFrame:
    """Deterministic strided pandas sample of the frame (empty-safe)."""
    indices = strided_sample_indices(df.height, max_rows)
    sampled = df.gather(pl.Series(indices)) if df.height else df
    return sampled.to_pandas()


def validate_input_contract(df: pl.DataFrame, cfg: NumPrepConfig) -> dict[str, Any]:
    """Validate the ingested snapshot against the input contract.

    Returns a summary dict for the report. Raises :class:`InputContractError`
    with every violation aggregated when anything fails.
    """
    contract = cfg.validation.input
    if not contract.enabled:
        return {"enabled": False, "violations": 0, "checked_rows": int(df.height)}

    violations: list[str] = []
    columns = list(df.columns)

    missing = [c for c in contract.required_columns if c not in columns]
    if missing:
        violations.append(f"missing required columns: {missing}")

    if cfg.target.column is not None and cfg.target.column not in columns:
        violations.append(
            f"configured target column {cfg.target.column!r} is absent from the input"
        )

    default_ceiling = contract.max_null_rate.get("*")
    ceilings_checked = 0
    for name in columns:
        if name == ROW_INDEX_COL:
            continue
        ceiling = contract.max_null_rate.get(name, default_ceiling)
        if ceiling is None:
            continue
        ceilings_checked += 1
        rate = float(df[name].null_count()) / max(int(df.height), 1)
        if rate > ceiling:
            violations.append(f"column {name!r} null rate {rate:.4f} exceeds ceiling {ceiling:.4f}")

    for name in contract.expected_numeric:
        if name not in columns:
            violations.append(f"expected numeric column {name!r} is absent")
            continue
        if not df[name].dtype.is_numeric():
            violations.append(f"column {name!r} has dtype {df[name].dtype}, expected numeric")

    ranges_checked = sum(1 for c in columns if c in contract.ranges)
    if ranges_checked and df.height:
        sample = _sampled_pandas_frame(df, contract.sample_rows)
        schema = _build_pandera_schema(contract, columns)
        try:
            schema.validate(sample, lazy=True)
        except SchemaErrors as exc:
            failure_cases = exc.failure_cases
            for count, (_, row) in enumerate(failure_cases.iterrows()):
                if count >= _MAX_REPORTED_FAILURES:
                    violations.append(
                        f"... and {len(failure_cases) - _MAX_REPORTED_FAILURES} more "
                        "range/contract failures (see sample above)"
                    )
                    break
                column = row.get("column", "<unknown>")
                check = row.get("check", "<unknown>")
                value = row.get("failure_case", "<unknown>")
                violations.append(
                    f"range check failed on column {column!r} (check: {check}, value: {value!r})"
                )

    if violations:
        rendered = "\n".join(["input contract violations:"] + [f"  - {v}" for v in violations])
        raise InputContractError(rendered)

    return {
        "enabled": True,
        "violations": 0,
        "checked_rows": int(df.height),
        "sampled_rows": min(int(df.height), contract.sample_rows) if ranges_checked else 0,
        "ceilings_checked": ceilings_checked,
        "ranges_checked": ranges_checked,
    }


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def _is_float_dtype(dtype: np.dtype | pd.api.extensions.ExtensionDtype) -> bool:
    """True for float dtypes (integers never hold inf)."""
    return bool(pd.api.types.is_float_dtype(dtype))


def validate_output_contract(
    partitions: dict[str, pd.DataFrame], target_column: str | None, cfg: NumPrepConfig
) -> dict[str, Any]:
    """Validate produced train/val/test partitions; raise on any violation.

    Checks (each config-gated): non-empty splits, identical column order and
    dtypes across splits, no NaN/inf, numeric-only feature columns, target
    present when configured.
    """
    out_cfg = cfg.validation.output
    violations: list[str] = []

    expected_names = ["train", "val", "test"]
    for name in expected_names:
        if name not in partitions:
            violations.append(f"partition {name!r} is missing from outputs")
    if violations:
        raise OutputContractError(
            "\n".join(["output contract violations:"] + [f"  - {v}" for v in violations])
        )

    if out_cfg.require_nonempty_splits:
        for name, frame in partitions.items():
            if frame.empty:
                violations.append(f"partition {name!r} is empty")

    reference = partitions["train"]
    if out_cfg.consistent_schema:
        ref_columns = list(reference.columns)
        ref_dtypes = {c: str(reference[c].dtype) for c in ref_columns}
        for name, frame in partitions.items():
            if name == "train":
                continue
            cols = list(frame.columns)
            if cols != ref_columns:
                first_diff = next(
                    (a for a, b in zip(cols, ref_columns, strict=False) if a != b), None
                )
                violations.append(
                    f"partition {name!r} column order differs from train (first diff: {first_diff})"
                )
                continue
            for col in ref_columns:
                if str(frame[col].dtype) != ref_dtypes[col]:
                    violations.append(
                        f"partition {name!r} column {col!r} dtype {frame[col].dtype} "
                        f"differs from train ({reference[col].dtype})"
                    )

    if target_column is not None:
        for name, frame in partitions.items():
            if target_column not in frame.columns:
                violations.append(f"partition {name!r} lacks target column {target_column!r}")

    if out_cfg.forbid_nan_inf:
        for name, frame in partitions.items():
            for col in frame.columns:
                if target_column is not None and col == target_column:
                    # The target is passed through untouched; flag NaN there too,
                    # but as a distinct, actionable message.
                    series = frame[col]
                    if pd.api.types.is_numeric_dtype(series) and bool(series.isna().any()):
                        violations.append(f"partition {name!r} target column {col!r} contains NaN")
                    continue
                series = frame[col]
                if not pd.api.types.is_numeric_dtype(series):
                    if out_cfg.require_numeric_features:
                        violations.append(
                            f"partition {name!r} feature column {col!r} is non-numeric "
                            f"({series.dtype}); enable preprocessing.encoding or drop it"
                        )
                    continue
                nan_mask = series.isna()
                inf_count = 0
                if _is_float_dtype(series.dtype):
                    inf_count = int(
                        np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).sum()
                    )
                if bool(nan_mask.any()) or inf_count:
                    violations.append(
                        f"partition {name!r} column {col!r} contains "
                        f"{int(nan_mask.sum())} NaN and {inf_count} inf values"
                    )

    if violations:
        rendered = "\n".join(["output contract violations:"] + [f"  - {v}" for v in violations])
        raise OutputContractError(rendered)

    return {
        "partitions": {name: len(frame) for name, frame in partitions.items()},
        "columns": len(reference.columns),
        "checks": {
            "forbid_nan_inf": out_cfg.forbid_nan_inf,
            "consistent_schema": out_cfg.consistent_schema,
            "require_nonempty_splits": out_cfg.require_nonempty_splits,
            "require_numeric_features": out_cfg.require_numeric_features,
        },
    }
