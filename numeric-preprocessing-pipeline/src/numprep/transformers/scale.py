"""Per-column scaling with a skew-aware ``auto`` mode; decisions recorded with reasons.

``auto`` selection (train-fitted, recorded per column):

* near-constant (σ < 1e-12 or ≤ 1 unique value) → ``none`` (passthrough),
* ``|skew| ≥ skew_high`` (default 1.0) → ``power`` (Yeo–Johnson — handles
  negatives and zeros, unlike a plain log),
* ``|skew| ≥ skew_moderate`` (default 0.5) → ``robust``,
* otherwise → ``standard``.

Config ``overrides`` pin exact columns to a kind and always win. Columns whose
names match ``exclude_patterns`` (one-hot ``cat_``-prefixed and missing
indicators by default) are passed through unscaled. The target never enters
this transformer — the orchestrator pops it before the pipeline.
"""

from __future__ import annotations

import re
from typing import Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from numprep.transformers.base import FrameTransformer

_SubScaler = Union[
    "StandardScaler", "RobustScaler", "MinMaxScaler", "PowerTransformer", "QuantileTransformer"
]

_KINDS = ("auto", "standard", "robust", "minmax", "power", "quantile", "none")
_NEAR_CONSTANT_STD = 1e-12


class GroupedScaler(FrameTransformer):
    """Fit one scaler per column according to config or the skew-aware auto rule."""

    def __init__(
        self,
        default: str = "auto",
        overrides: dict[str, str] | None = None,
        exclude_patterns: tuple[str, ...] = (r"^cat_", r"__was_missing$"),
        skew_high: float = 1.0,
        skew_moderate: float = 0.5,
        quantile_output_distribution: str = "normal",
        quantile_n_quantiles: int = 1000,
    ) -> None:
        self.default = default
        self.overrides = overrides  # stored unmodified: sklearn clone contract
        self.exclude_patterns = tuple(exclude_patterns)
        self.skew_high = skew_high
        self.skew_moderate = skew_moderate
        self.quantile_output_distribution = quantile_output_distribution
        self.quantile_n_quantiles = quantile_n_quantiles

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_kind(self, name: str, series: pd.Series) -> tuple[str, str]:
        """Return (kind, reason) for one column."""
        overrides = self.overrides or {}
        if name in overrides:
            return overrides[name], "config override (preprocessing.scaling.overrides)"
        for pattern in self.exclude_patterns:
            if re.search(pattern, name):
                return "none", f"excluded by scaling.exclude_patterns {pattern!r}"
        if not pd.api.types.is_numeric_dtype(series):
            return "none", "non-numeric column passed through"
        if self.default != "auto":
            return self.default, f"configured default {self.default!r}"

        values = series.to_numpy(dtype="float64")
        finite = values[~np.isnan(values)]
        if finite.size < 3:
            return "none", "fewer than 3 finite values on train"
        std = float(np.std(finite))
        if std <= _NEAR_CONSTANT_STD or np.unique(finite).size <= 1:
            return "none", "near-constant on train (no scaling information)"
        # pandas skew (Fisher-Pearson), matching the report's skew definition.
        skew_value = pd.Series(finite).skew()
        # pandas' .skew() is typed loosely; the value is a float when not NaN.
        skew = 0.0 if pd.isna(skew_value) else float(skew_value)  # type: ignore[arg-type]
        magnitude = abs(skew) if not np.isnan(skew) else 0.0
        if magnitude >= self.skew_high:
            return (
                "power",
                f"auto: |skew|={magnitude:.3f} >= skew_high {self.skew_high:g} → Yeo-Johnson",
            )
        if magnitude >= self.skew_moderate:
            return (
                "robust",
                f"auto: |skew|={magnitude:.3f} >= skew_moderate {self.skew_moderate:g} → robust",
            )
        return (
            "standard",
            f"auto: |skew|={magnitude:.3f} < skew_moderate {self.skew_moderate:g} → standard",
        )

    def _build_scaler(self, kind: str, n_rows: int) -> _SubScaler:
        """Instantiate the sklearn sub-scaler for a kind."""
        if kind == "standard":
            return StandardScaler()
        if kind == "robust":
            return RobustScaler()
        if kind == "minmax":
            return MinMaxScaler()
        if kind == "power":
            return PowerTransformer(method="yeo-johnson", standardize=True)
        if kind == "quantile":
            return QuantileTransformer(
                n_quantiles=min(self.quantile_n_quantiles, max(n_rows, 1)),
                output_distribution=self.quantile_output_distribution,
                subsample=1_000_000,
                random_state=0,
            )
        raise AssertionError(f"unreachable scaler kind {kind!r}")  # guarded by _decide_kind

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> GroupedScaler:
        """Decide and fit one sub-scaler per column (train only)."""
        if self.default not in _KINDS:
            raise ValueError(f"unknown scaling default {self.default!r}; expected one of {_KINDS}")
        X = self._check_input(X)
        unknown_overrides = set(self.overrides or {}) - set(X.columns)
        if unknown_overrides:
            raise ValueError(
                f"preprocessing.scaling.overrides reference unknown columns: "
                f"{sorted(unknown_overrides)}"
            )
        self._columns_in_ = list(X.columns)
        self.kinds_: dict[str, str] = {}
        self.reasons_: dict[str, str] = {}
        self.scalers_: dict[str, _SubScaler] = {}

        for name in self._columns_in_:
            kind, reason = self._decide_kind(name, X[name])
            self.kinds_[name] = kind
            self.reasons_[name] = reason
            if kind != "none":
                scaler = self._build_scaler(kind, len(X))
                scaler.fit(X[[name]].to_numpy(dtype="float64"))
                self.scalers_[name] = scaler

        self.feature_names_out_ = list(self._columns_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply each column's fitted scaler; excluded/passthrough columns unchanged."""
        if not hasattr(self, "kinds_"):
            raise ValueError("This GroupedScaler instance is not fitted yet; call fit() first.")
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")

        result = X.copy(deep=False)
        for name in self._columns_in_:
            scaler = self.scalers_.get(name)
            if scaler is None:
                continue
            column_values = X[[name]].to_numpy(dtype="float64")
            with np.errstate(invalid="ignore"):
                transformed = cast(np.ndarray, scaler.transform(column_values))
            result[name] = transformed.reshape(-1)
        if len(result) != len(X):  # pragma: no cover - defensive
            raise ValueError("scaler changed the row count")
        return result
