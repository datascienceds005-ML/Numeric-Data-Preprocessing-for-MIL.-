"""Outlier handling: train-fitted bounds, in-place clipping, never row drops.

Bounds are learned on train only and applied identically to val/test — evaluation
rows are clipped, never dropped, and every clipped value is counted and logged.
Methods:

* ``winsorize`` — clip to the ``lower_quantile``/``upper_quantile`` percentiles,
* ``iqr`` — clip to ``Q1 - k·IQR`` / ``Q3 + k·IQR``,
* ``zscore`` — clip to ``mean ± cap·std`` (constant columns skipped),
* ``none`` — leave values untouched.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from numprep.jsonlog import log_event
from numprep.transformers.base import FrameTransformer

logger = logging.getLogger("numprep")


class OutlierClipper(FrameTransformer):
    """Clip numeric features to train-fitted bounds.

    Parameters
    ----------
    method:
        ``winsorize`` / ``iqr`` / ``zscore`` / ``none``.
    lower_quantile, upper_quantile:
        Percentile bounds for ``winsorize``.
    iqr_k:
        IQR multiplier for the ``iqr`` method.
    zscore_cap:
        Standard-deviation cap for the ``zscore`` method.
    """

    def __init__(
        self,
        method: str = "winsorize",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_k: float = 1.5,
        zscore_cap: float = 3.0,
    ) -> None:
        self.method = method
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.iqr_k = iqr_k
        self.zscore_cap = zscore_cap

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> OutlierClipper:
        """Learn per-column clip bounds from train only."""
        if self.method not in ("winsorize", "iqr", "zscore", "none"):
            raise ValueError(f"unknown outlier method {self.method!r}")
        if self.method == "winsorize" and self.lower_quantile > self.upper_quantile:
            raise ValueError("lower_quantile must be <= upper_quantile")
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        self.bounds_: dict[str, tuple[float, float] | None] = {}

        for name in self._columns_in_:
            if not pd.api.types.is_numeric_dtype(X[name]):
                self.bounds_[name] = None
                continue
            values = X[name].to_numpy(dtype="float64")
            finite = values[~np.isnan(values)]
            if finite.size == 0 or self.method == "none":
                self.bounds_[name] = None
                continue
            if self.method == "winsorize":
                low, high = np.quantile(finite, [self.lower_quantile, self.upper_quantile])
            elif self.method == "iqr":
                q1, q3 = np.quantile(finite, [0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
            else:  # zscore
                std = float(np.std(finite))
                if std == 0.0:
                    self.bounds_[name] = None
                    continue
                mean = float(np.mean(finite))
                low, high = mean - self.zscore_cap * std, mean + self.zscore_cap * std
            self.bounds_[name] = (float(low), float(high))

        self.feature_names_out_ = list(self._columns_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip in place; NaN preserved; clipped counts logged per column."""
        if not hasattr(self, "bounds_"):
            raise ValueError("This OutlierClipper instance is not fitted yet; call fit() first.")
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")

        clipped_counts: dict[str, int] = {}
        active = {k: v for k, v in self.bounds_.items() if v is not None}
        if not active:
            return X

        result = X.copy(deep=False)
        for name, (low, high) in active.items():
            series = result[name]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            numeric = series.astype("float64")
            outside = ((numeric < low) | (numeric > high)) & numeric.notna()
            count = int(outside.sum())
            if count:
                clipped_counts[name] = count
                result[name] = numeric.clip(lower=low, upper=high)

        if clipped_counts:
            log_event(
                logger,
                logging.INFO,
                "outlier_clip_applied",
                partition_tag=X.attrs.get("numprep.partition", "unknown"),
                clipped_values=clipped_counts,
                total_clipped=sum(clipped_counts.values()),
            )
            self.last_transform_stats_ = clipped_counts
        if len(result) != len(X):  # pragma: no cover - defensive
            raise ValueError("clipper changed the row count")
        return result
