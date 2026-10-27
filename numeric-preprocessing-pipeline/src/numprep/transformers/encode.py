"""One-hot encoding of low-cardinality categorical/boolean columns.

Behavior (all decisions fitted on train):

* object / bool / category columns with ≤ ``max_cardinality`` train categories
  are one-hot encoded as ``cat_<col>__<value>`` float32 columns;
* the category list is truncated to the ``onehot_max_categories`` most frequent
  values (ties broken by value, deterministic);
* **unseen-category safety:** values absent from the fitted list (including
  nulls) map to an all-zero row — never an error, never a NaN;
* high-cardinality string columns are dropped with a recorded warning when
  ``drop_high_cardinality`` is true (otherwise they surface later as an
  output-contract violation, which is the honest failure mode);
* the ``cat_`` prefix doubles as the default scaling-exclusion pattern, so
  one-hots are never standardised.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from numprep.jsonlog import log_event
from numprep.transformers.base import FrameTransformer

_ENCODE_PREFIX = "cat_"
_CATEGORICAL_KINDS = frozenset({"O", "b"})
_ONEHOT_DTYPE = "float32"

logger = logging.getLogger("numprep")


class LowCardinalityEncoder(FrameTransformer):
    """Encode low-cardinality categorical/boolean columns with unseen safety."""

    def __init__(
        self,
        max_cardinality: int = 10,
        onehot_max_categories: int = 25,
        drop_high_cardinality: bool = True,
    ) -> None:
        self.max_cardinality = max_cardinality
        self.onehot_max_categories = onehot_max_categories
        self.drop_high_cardinality = drop_high_cardinality

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> LowCardinalityEncoder:
        """Choose which columns to encode, drop, or pass through (train-fitted)."""
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        self.categories_: dict[str, list[str]] = {}
        self.dropped_columns_: dict[str, str] = {}
        self.encoded_columns_: list[str] = []
        self.expanded_names_: list[str] = []

        for name in self._columns_in_:
            series = X[name]
            kind = series.dtype.kind
            is_category = kind in _CATEGORICAL_KINDS or str(series.dtype) == "category"
            if not is_category:
                continue
            string_values = series.astype(str).where(series.notna(), other=None)
            unique_count = int(string_values.dropna().nunique())
            if unique_count > self.max_cardinality:
                if self.drop_high_cardinality:
                    self.dropped_columns_[name] = (
                        f"{unique_count} train categories > max_cardinality={self.max_cardinality}"
                    )
                continue
            if unique_count == 0:
                self.dropped_columns_[name] = "no non-null values on train"
                continue
            frequencies = string_values.dropna().value_counts()
            ordered = sorted(
                frequencies.index.tolist(),
                key=lambda value: (-int(frequencies[value]), value),
            )[: self.onehot_max_categories]
            ordered = sorted(ordered)  # stable, deterministic column order
            self.categories_[name] = [str(value) for value in ordered]
            self.encoded_columns_.append(name)
            self.expanded_names_.extend(
                f"{_ENCODE_PREFIX}{name}__{value}" for value in self.categories_[name]
            )

        self.feature_names_out_ = [
            name for name in self._columns_in_ if name not in self.dropped_columns_
        ]
        self.feature_names_out_ = _expand(self.feature_names_out_, self.categories_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """One-hot fitted columns; unseen values become all-zero rows."""
        if not hasattr(self, "categories_"):
            raise ValueError(
                "This LowCardinalityEncoder instance is not fitted yet; call fit() first."
            )
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        if not self.encoded_columns_ and not self.dropped_columns_:
            return X

        unseen_counts: dict[str, int] = {}
        blocks: dict[str, pd.DataFrame] = {}
        for name, categories in self.categories_.items():
            series = X[name]
            isna = series.isna()
            as_str = series.astype(str).where(~isna, other=None)
            known = as_str.isin(categories)
            unseen = int((~known & ~isna).sum())
            if unseen:
                unseen_counts[name] = unseen
            categorical = pd.Categorical(as_str, categories=categories)
            codes = categorical.codes.astype(np.int64)
            matrix = np.zeros((len(categorical), len(categories)), dtype="float32")
            valid = codes >= 0
            matrix[valid, codes[valid]] = 1.0
            blocks[name] = pd.DataFrame(
                matrix,
                index=X.index,
                columns=[f"{_ENCODE_PREFIX}{name}__{value}" for value in categories],
                dtype=_ONEHOT_DTYPE,
            )

        parts: list[pd.DataFrame] = []
        for name in self._columns_in_:
            if name in self.dropped_columns_:
                continue
            if name in blocks:
                parts.append(blocks[name])
            else:
                parts.append(X[[name]])
        out = pd.concat(parts, axis=1)
        out = out[self.feature_names_out_]

        if unseen_counts:
            log_event(
                logger,
                logging.INFO,
                "encoder_unseen_categories_zeroed",
                unseen_counts=unseen_counts,
                note="unseen categories map to all-zero one-hot rows",
            )
        if len(out) != len(X):  # pragma: no cover - defensive
            raise ValueError("encoder changed the row count")
        return out


def _expand(columns: list[str], categories: dict[str, list[str]]) -> list[str]:
    """Expand fitted column names with their one-hot expansions, in order."""
    out: list[str] = []
    for name in columns:
        if name in categories:
            out.extend(f"{_ENCODE_PREFIX}{name}__{value}" for value in categories[name])
        else:
            out.append(name)
    return out
