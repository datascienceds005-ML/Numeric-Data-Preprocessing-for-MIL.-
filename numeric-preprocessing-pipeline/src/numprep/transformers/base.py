"""Shared base class for DataFrame-in / DataFrame-out transformers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from numprep.errors import PipelineError


class FrameTransformer(BaseEstimator, TransformerMixin):
    """Base class for all numprep transformers.

    Provides:

    * strict input validation (must be a pandas ``DataFrame`` without duplicated
      column names),
    * column-presence validation against the fitted signature,
    * a row-count-preserving ``DataFrame`` builder (rows are never silently
      dropped after the split — a hard project rule),
    * a ``get_feature_names_out`` default backed by ``feature_names_out_``.
    """

    #: Fitted by every subclass at the end of ``fit``; drives feature-name APIs.
    feature_names_out_: list[str]

    def _check_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate the input frame type and column-name uniqueness."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{type(self).__name__} expects a pandas DataFrame, got {type(X).__name__}"
            )
        duplicated = X.columns[X.columns.duplicated()].tolist()
        if duplicated:
            raise ValueError(
                f"{type(self).__name__} received duplicated column names: {duplicated}"
            )
        return X

    def _require_fitted_columns(self, X: pd.DataFrame, columns: Iterable[str], stage: str) -> None:
        """Raise an actionable error when columns seen at fit are missing at transform."""
        missing = [name for name in columns if name not in X.columns]
        if missing:
            raise ValueError(
                f"{type(self).__name__} ({stage}) is missing columns present at fit "
                f"time: {missing}. Ensure all partitions share the input schema."
            )

    def _build_output(
        self, X: pd.DataFrame, values: np.ndarray | pd.DataFrame, columns: Sequence[str]
    ) -> pd.DataFrame:
        """Wrap transformed values into a DataFrame aligned with the input index."""
        if len(columns) != values.shape[1]:
            raise PipelineError(
                f"{type(self).__name__}: column count ({len(columns)}) does not match "
                f"transformed width ({values.shape[1]})"
            )
        out = pd.DataFrame(values, index=X.index, columns=list(columns), copy=False)
        if len(out) != len(X):
            raise PipelineError(
                f"{type(self).__name__} changed the row count "
                f"({len(X)} -> {len(out)}); rows must never be dropped or added "
                "after the split"
            )
        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return output column names; ``input_features`` is validated when given."""
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None:
            unexpected = [name for name in input_features if name not in self._columns_in_]
            if unexpected:
                raise ValueError(
                    f"unexpected input features {unexpected}; fitted columns were "
                    f"{list(self._columns_in_)}"
                )
        return np.asarray(self.feature_names_out_, dtype=object)
