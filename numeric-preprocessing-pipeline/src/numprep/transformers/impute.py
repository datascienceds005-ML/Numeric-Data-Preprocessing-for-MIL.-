"""Feature imputation (numeric columns only; the target is never imputed).

Strategies
----------
* ``median`` (default — robust to the outliers this pipeline deliberately keeps),
* ``mean``,
* ``constant`` (``constant_value``),
* ``knn`` — :class:`sklearn.impute.KNNImputer` (deterministic; multivariate),
* ``iterative`` — :class:`sklearn.experimental`'s ``IterativeImputer``
  (``random_state`` = run seed; opt-in because it is the slowest option).

Non-numeric columns pass through untouched (the fitted encoder handles them
downstream). An all-null column cannot be imputed by any strategy and surfaces
as an output-contract failure with an actionable message — the honest outcome,
rather than a silently fabricated feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (needed import)
from sklearn.impute import IterativeImputer, KNNImputer

from numprep.transformers.base import FrameTransformer

_STRATEGIES = ("median", "mean", "constant", "knn", "iterative")


class NumPrepImputer(FrameTransformer):
    """Impute numeric feature columns with a configurable strategy.

    Parameters
    ----------
    strategy:
        One of ``median`` / ``mean`` / ``constant`` / ``knn`` / ``iterative``.
    constant_value:
        Fill value for the ``constant`` strategy.
    knn_neighbors:
        Neighbours for the ``knn`` strategy.
    iterative_max_iter:
        Max iterations for the ``iterative`` strategy.
    random_state:
        Seed for the ``iterative`` strategy (wired to the global run seed).
    """

    def __init__(
        self,
        strategy: str = "median",
        constant_value: float = 0.0,
        knn_neighbors: int = 5,
        iterative_max_iter: int = 10,
        random_state: int | None = None,
    ) -> None:
        self.strategy = strategy
        self.constant_value = constant_value
        self.knn_neighbors = knn_neighbors
        self.iterative_max_iter = iterative_max_iter
        self.random_state = random_state

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> NumPrepImputer:
        """Learn per-column fill statistics (or fit a multivariate sub-imputer)."""
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"unknown imputation strategy {self.strategy!r}; expected one of {_STRATEGIES}"
            )
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        self.numeric_columns_ = [
            name for name in X.columns if pd.api.types.is_numeric_dtype(X[name])
        ]

        if self.strategy in ("knn", "iterative"):
            matrix = X[self.numeric_columns_].to_numpy(dtype="float64")
            multivariate: KNNImputer | IterativeImputer
            if self.strategy == "knn":
                multivariate = KNNImputer(n_neighbors=self.knn_neighbors).fit(matrix)
                self.statistics_ = {
                    name: float(np.nanmedian(X[name].to_numpy(dtype="float64")))
                    for name in self.numeric_columns_
                }
            else:
                multivariate = IterativeImputer(
                    max_iter=self.iterative_max_iter,
                    random_state=self.random_state,
                ).fit(matrix)
                self.statistics_ = {
                    name: float(value)
                    for name, value in zip(
                        self.numeric_columns_,
                        multivariate.initial_imputer_.statistics_,
                        strict=False,
                    )
                }
            self.multivariate_imputer_ = multivariate
        else:
            self.statistics_ = {}
            for name in self.numeric_columns_:
                values = X[name].to_numpy(dtype="float64")
                if self.strategy == "median":
                    fill = float(np.nanmedian(values)) if (~np.isnan(values)).any() else np.nan
                elif self.strategy == "mean":
                    fill = float(np.nanmean(values)) if (~np.isnan(values)).any() else np.nan
                else:
                    fill = float(self.constant_value)
                self.statistics_[name] = fill

        self.feature_names_out_ = list(self._columns_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric nulls; non-numeric columns pass through unchanged."""
        if not hasattr(self, "statistics_"):
            raise ValueError("This NumPrepImputer instance is not fitted yet; call fit() first.")
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        if not self.numeric_columns_ or not any(
            bool(X[name].isna().any()) for name in self.numeric_columns_
        ):
            return X

        if self.strategy in ("knn", "iterative"):
            matrix = X[self.numeric_columns_].to_numpy(dtype="float64")
            imputed = self.multivariate_imputer_.transform(matrix)
            result = X.copy(deep=False)
            for position, name in enumerate(self.numeric_columns_):
                result[name] = imputed[:, position]
            if len(result) != len(X):  # pragma: no cover - defensive
                raise ValueError("imputer changed the row count")
            return result

        filled = X.copy(deep=False)
        for name in self.numeric_columns_:
            if bool(filled[name].isna().any()):
                filled[name] = filled[name].fillna(self.statistics_[name])
        if len(filled) != len(X):  # pragma: no cover - defensive
            raise ValueError("imputer changed the row count")
        return filled
