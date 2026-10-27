"""Train-fitted feature selection: variance threshold + greedy correlation pruning.

* Variance: features with train variance ``<= variance_threshold`` are dropped
  (threshold ``0`` therefore removes exactly-constant features, mirroring
  sklearn's ``VarianceThreshold``).
* Correlation: on a deterministic strided sample, the pair with the highest
  absolute Pearson correlation above ``correlation_threshold`` loses the member
  with the higher mean absolute correlation to the remaining features (ties:
  lexicographically larger name goes). The loop repeats until no surviving pair
  exceeds the threshold. Deterministic end-to-end; every drop records its
  partner and the correlation value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numprep.sampling import strided_sample_indices
from numprep.transformers.base import FrameTransformer


class FeatureSelector(FrameTransformer):
    """Variance + correlation feature selection, fitted on train only."""

    def __init__(
        self,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.98,
        sample_size: int = 200_000,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.sample_size = sample_size

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureSelector:
        """Learn the drop set from train statistics."""
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        numeric = [name for name in self._columns_in_ if pd.api.types.is_numeric_dtype(X[name])]
        drop_reasons: dict[str, str] = {}

        if numeric:
            variances = X[numeric].var(skipna=True)
            for name in numeric:
                variance = float(variances[name])
                if pd.isna(variance) or variance <= self.variance_threshold:
                    drop_reasons[name] = (
                        f"train variance {variance:.6g} <= threshold {self.variance_threshold:g}"
                    )

        corr_drops = self._correlation_pruning(X, numeric)
        for name, reason in corr_drops.items():
            if name not in drop_reasons:
                drop_reasons[name] = reason

        self.drop_reasons_ = drop_reasons
        self.columns_to_keep_ = [c for c in self._columns_in_ if c not in drop_reasons]
        self.feature_names_out_ = list(self.columns_to_keep_)
        return self

    def _correlation_pruning(self, X: pd.DataFrame, numeric: list[str]) -> dict[str, str]:
        """Greedy high-correlation pruning on a deterministic sample."""
        survivors = list(numeric)
        if len(survivors) < 2:
            return {}
        indices = strided_sample_indices(len(X), self.sample_size)
        sample = X[survivors].iloc[indices]
        corr = sample.corr(method="pearson").to_numpy(dtype="float64")
        names = list(survivors)
        alive = [True] * len(names)
        drops: dict[str, str] = {}
        abs_corr = np.abs(np.nan_to_num(corr, nan=0.0))
        mean_abs = abs_corr.sum(axis=1) / max(len(names) - 1, 1)

        while True:
            best_score = 0.0
            best_pair: tuple[int, int] | None = None
            for i in range(len(names)):
                if not alive[i]:
                    continue
                for j in range(i + 1, len(names)):
                    if not alive[j]:
                        continue
                    score = abs(corr[i, j]) if not np.isnan(corr[i, j]) else 0.0
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair is None or best_score <= self.correlation_threshold:
                break
            i, j = best_pair
            # Drop the more-connected member; ties drop the lexicographically larger.
            if (mean_abs[i], names[i]) > (mean_abs[j], names[j]):
                victim, keeper = i, j
            else:
                victim, keeper = j, i
            alive[victim] = False
            drops[names[victim]] = (
                f"|r|={best_score:.4f} with {names[keeper]} > "
                f"threshold {self.correlation_threshold}; higher mean |r| "
                f"({mean_abs[victim]:.4f} vs {mean_abs[keeper]:.4f})"
            )
            # Recompute mean |r| against surviving columns only.
            for k in range(len(names)):
                if alive[k]:
                    others = [m for m in range(len(names)) if alive[m] and m != k]
                    if others:
                        mean_abs[k] = float(np.mean([abs_corr[k, m] for m in others]))
                    else:
                        mean_abs[k] = 0.0
        return drops

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep the fitted columns in fitted order."""
        if not hasattr(self, "columns_to_keep_"):
            raise ValueError("This FeatureSelector instance is not fitted yet; call fit() first.")
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        kept = X[self.columns_to_keep_]
        if len(kept) != len(X):  # pragma: no cover - defensive
            raise ValueError("selector changed the row count")
        return kept
