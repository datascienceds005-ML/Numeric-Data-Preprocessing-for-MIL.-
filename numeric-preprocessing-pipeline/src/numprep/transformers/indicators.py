"""Missing-indicator columns added before imputation.

For every train column containing nulls, a ``f"{col}__was_missing"`` binary flag
records *where* the imputer intervened — often itself a predictive signal. The
set of indicator columns (and their order) is fitted on train; partitions that
somehow lack a flagged column fail loudly rather than drifting silently.
"""

from __future__ import annotations

import pandas as pd

from numprep.transformers.base import FrameTransformer

_INDICATOR_SUFFIX = "__was_missing"


class MissingIndicatorAdder(FrameTransformer):
    """Add train-fitted missing-indicator columns.

    Parameters
    ----------
    max_indicators:
        Cap on the number of indicator columns. Selection is deterministic:
        columns with the most train nulls first, ties broken by column name.
        ``0`` disables the feature entirely.
    """

    def __init__(self, max_indicators: int = 50) -> None:
        self.max_indicators = max_indicators

    def fit(self, X: pd.DataFrame, y: object = None) -> MissingIndicatorAdder:
        """Select the columns that will receive indicator flags on train."""
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        null_counts = X.isna().sum()
        candidates = sorted(
            (name for name in X.columns if null_counts[name] > 0),
            key=lambda name: (-int(null_counts[name]), name),
        )
        if self.max_indicators >= 0:
            candidates = candidates[: self.max_indicators]
        self.indicator_columns_ = candidates
        self.feature_names_out_ = self._columns_in_ + [
            f"{name}{_INDICATOR_SUFFIX}" for name in candidates
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append indicator columns (uint8) in the fitted order."""
        if not hasattr(self, "indicator_columns_"):
            raise ValueError(
                "This MissingIndicatorAdder instance is not fitted yet; call fit() first."
            )
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        if not self.indicator_columns_:
            return X
        indicator_frame = pd.DataFrame(
            {
                f"{name}{_INDICATOR_SUFFIX}": X[name].isna().astype("uint8").to_numpy()
                for name in self.indicator_columns_
            },
            index=X.index,
        )
        out = pd.concat([X, indicator_frame], axis=1)
        out = out[self.feature_names_out_]
        if len(out) != len(X):  # pragma: no cover - defensive
            raise ValueError("indicator adder changed the row count")
        return out
