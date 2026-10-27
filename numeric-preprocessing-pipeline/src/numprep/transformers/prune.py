"""Train-fitted column pruning: null-rate ceiling, constants, ID-like columns.

The decision to drop a column is a *statistic* and therefore fitted on the train
partition only (DESIGN §4.2). The pruner is deliberately the first pipeline step
so every later step sees a stable schema, and it double-checks the partition tag
itself (defense in depth behind ``pipeline.fit_preprocessor``).
"""

from __future__ import annotations

import re

import pandas as pd

from numprep.errors import LeakageError
from numprep.partitions import Partition, partition_of
from numprep.transformers.base import FrameTransformer

_NUMERIC_INTEGER_KINDS = frozenset({"i", "u"})


class TrainOnlyColumnPruner(FrameTransformer):
    """Drop columns whose *train* statistics justify removal.

    Parameters
    ----------
    null_threshold:
        Drop columns with a train null-rate strictly above this value.
    drop_constant:
        Drop columns with zero variance on train.
    drop_id_like:
        Drop ID-like columns: name matches ``id_name_pattern`` **or** the column
        is integer/string typed with a train unique-ratio ≥ ``id_unique_ratio``.
    excluded:
        Columns that must never be dropped (reserved for provenance keys).
    """

    def __init__(
        self,
        null_threshold: float = 0.95,
        drop_constant: bool = True,
        drop_id_like: bool = True,
        id_name_pattern: str = r"(?i)^(id|.*[_-]id|uuid|guid|index|row_?num|.*[_-]num)$",
        id_unique_ratio: float = 0.999,
        excluded: tuple[str, ...] = (),
    ) -> None:
        self.null_threshold = null_threshold
        self.drop_constant = drop_constant
        self.drop_id_like = drop_id_like
        self.id_name_pattern = id_name_pattern
        self.id_unique_ratio = id_unique_ratio
        self.excluded = tuple(excluded)

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> TrainOnlyColumnPruner:
        """Learn drop decisions from the train partition only."""
        if partition_of(X) is not Partition.TRAIN:
            raise LeakageError(
                "TrainOnlyColumnPruner.fit requires the tagged train partition; "
                f"received {partition_of(X)!r}"
            )
        X = self._check_input(X)
        name_re = re.compile(self.id_name_pattern)
        self._columns_in_ = list(X.columns)
        drop_reasons: dict[str, str] = {}

        for column in X.columns:
            if column in self.excluded:
                continue
            series = X[column]
            null_rate = float(series.isna().mean())
            if null_rate > self.null_threshold:
                drop_reasons[column] = (
                    f"train null rate {null_rate:.4f} > threshold {self.null_threshold}"
                )
                continue
            if self.drop_constant and series.nunique(dropna=False) <= 1:
                drop_reasons[column] = "constant on train (zero variance)"
                continue
            if self.drop_id_like and self._is_id_like(name_re, series):
                drop_reasons[column] = (
                    f"id-like: name matches {self.id_name_pattern!r} or train "
                    f"unique ratio >= {self.id_unique_ratio}"
                )

        self.drop_reasons_ = drop_reasons
        self.columns_to_keep_ = [c for c in self._columns_in_ if c not in drop_reasons]
        self.feature_names_out_ = list(self.columns_to_keep_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the fitted columns; output preserves the input row order."""
        check_self = getattr(self, "columns_to_keep_", None)
        if check_self is None:
            raise ValueError(
                "This TrainOnlyColumnPruner instance is not fitted yet; call fit() first."
            )
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        kept = X[self.columns_to_keep_]
        if len(kept) != len(X):  # pragma: no cover - defensive
            raise ValueError("pruner changed the row count")
        return kept

    # Helpers ---------------------------------------------------------------

    def _is_id_like(self, name_re: re.Pattern[str], series: pd.Series) -> bool:
        """Name-pattern hit, or integer/string column with ~all-unique values."""
        if name_re.search(str(series.name)):
            return True
        kind = series.dtype.kind
        if kind in _NUMERIC_INTEGER_KINDS or series.dtype == object:
            unique_ratio = float(series.nunique(dropna=True)) / max(len(series), 1)
            return unique_ratio >= self.id_unique_ratio and series.nunique(dropna=True) > 10
        return False
