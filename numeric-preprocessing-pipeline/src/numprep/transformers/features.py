"""Config-driven feature engineering with a generated, auditable catalog.

Supported operations (each a spec in ``preprocessing.features``):

* ``log1p`` — ``log1p(x)`` when the train minimum is ≥ 0; the **signed** variant
  ``sign(x)·log1p(|x|)`` when negatives are present (always finite for finite x).
* ``ratios`` — ``numerator / denominator`` with safe division: a denominator with
  ``|d| < safe_division_eps`` (or zero / NaN) yields NaN, which is then replaced
  by the *train median of the engineered feature* — no inf ever escapes.
* ``differences`` — ``minuend - subtrahend``.
* ``interactions`` — ``left * right``.
* ``polynomial`` — ``column ** degree`` (integer degrees only; finite-checked).

Engineered outputs that end up entirely null/inf on train are dropped with a
recorded reason instead of poisoning downstream stages. Every generated feature
carries a catalog entry: name, formula, and a one-line rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from numprep.errors import PipelineError
from numprep.transformers.base import FrameTransformer


@dataclass(frozen=True)
class RatioSpec:
    """``name = numerator / denominator`` with safe division."""

    numerator: str
    denominator: str
    name: str | None = None


@dataclass(frozen=True)
class DiffSpec:
    """``name = minuend - subtrahend``."""

    minuend: str
    subtrahend: str
    name: str | None = None


@dataclass(frozen=True)
class InteractionSpec:
    """``name = left * right``."""

    left: str
    right: str
    name: str | None = None


@dataclass(frozen=True)
class PolySpec:
    """``name = column ** degree``."""

    column: str
    degree: int
    name: str | None = None


@dataclass(frozen=True)
class CatalogEntry:
    """One generated feature's documentation."""

    name: str
    kind: str
    formula: str
    rationale: str
    source_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class FeatureEngineer(FrameTransformer):
    """Generate engineered features from explicit specs; fill their NaN/inf on train.

    Parameters
    ----------
    log1p:
        Source columns for (signed) log1p features.
    ratios, differences, interactions, polynomial:
        Spec objects (importable from this module).
    safe_division_eps:
        Denominators with ``|value| < eps`` are treated as division by zero.
    max_engineered:
        Hard cap on generated features (guards against config blow-ups).
    """

    def __init__(
        self,
        log1p: tuple[str, ...] = (),
        ratios: tuple[RatioSpec, ...] = (),
        differences: tuple[DiffSpec, ...] = (),
        interactions: tuple[InteractionSpec, ...] = (),
        polynomial: tuple[PolySpec, ...] = (),
        safe_division_eps: float = 1e-12,
        max_engineered: int = 200,
    ) -> None:
        self.log1p = tuple(log1p)
        self.ratios = tuple(ratios)
        self.differences = tuple(differences)
        self.interactions = tuple(interactions)
        self.polynomial = tuple(polynomial)
        self.safe_division_eps = safe_division_eps
        self.max_engineered = max_engineered

    # ------------------------------------------------------------------
    # Spec → (name, formula, rationale) resolution
    # ------------------------------------------------------------------

    def _resolve_specs(self) -> list[tuple[str, str, str, str, list[str], str]]:
        """Flatten all spec kinds into (kind, name, formula, rationale, sources, key)."""
        spec: RatioSpec | DiffSpec | InteractionSpec | PolySpec
        specs: list[tuple[str, str, str, str, list[str], str]] = []
        for column in self.log1p:
            specs.append(
                (
                    "log1p",
                    f"{column}__log1p",
                    f"log1p({column})",
                    (
                        f"Compresses right-skew of {column!r}; variance-stabilising and "
                        "makes multiplicative effects additive."
                    ),
                    [column],
                    f"log1p:{column}",
                )
            )
        for spec in self.ratios:
            name = spec.name or f"{spec.numerator}__div__{spec.denominator}"
            specs.append(
                (
                    "ratio",
                    name,
                    f"{spec.numerator} / {spec.denominator} "
                    f"(|den| < {self.safe_division_eps:g} → NaN → train median)",
                    (
                        f"Normalises {spec.numerator!r} by {spec.denominator!r} to expose "
                        "relative magnitude; safe division avoids infinities."
                    ),
                    [spec.numerator, spec.denominator],
                    f"ratio:{spec.numerator}/{spec.denominator}",
                )
            )
        for spec in self.differences:
            name = spec.name or f"{spec.minuend}__minus__{spec.subtrahend}"
            specs.append(
                (
                    "difference",
                    name,
                    f"{spec.minuend} - {spec.subtrahend}",
                    f"Captures the change/imbalance between {spec.minuend!r} "
                    f"and {spec.subtrahend!r}.",
                    [spec.minuend, spec.subtrahend],
                    f"diff:{spec.minuend}-{spec.subtrahend}",
                )
            )
        for spec in self.interactions:
            name = spec.name or f"{spec.left}__x__{spec.right}"
            specs.append(
                (
                    "interaction",
                    name,
                    f"{spec.left} * {spec.right}",
                    f"Captures the joint multiplicative effect of {spec.left!r} "
                    f"and {spec.right!r}.",
                    [spec.left, spec.right],
                    f"inter:{spec.left}*{spec.right}",
                )
            )
        for spec in self.polynomial:
            name = spec.name or f"{spec.column}__pow{spec.degree}"
            specs.append(
                (
                    "polynomial",
                    name,
                    f"{spec.column} ** {spec.degree}",
                    (
                        f"Degree-{spec.degree} term of {spec.column!r} to expose "
                        "non-linear monotonic structure."
                    ),
                    [spec.column],
                    f"poly:{spec.column}^{spec.degree}",
                )
            )
        return specs

    def _validate_sources(self, X: pd.DataFrame, specs: list[tuple]) -> None:
        missing = sorted(
            {source for spec in specs for source in spec[4] if source not in X.columns}
        )
        if missing:
            raise PipelineError(f"feature engineering references absent columns: {missing}")

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute(self, kind: str, sources: list[str], key: str, X: pd.DataFrame) -> pd.Series:
        """Compute one engineered feature; inf/NaN may appear and are handled by fit/transform."""
        if kind == "log1p":
            column = sources[0]
            values = X[column].to_numpy(dtype="float64")
            finite_min = np.nanmin(values) if (~np.isnan(values)).any() else np.nan
            if np.isnan(finite_min) or finite_min >= 0.0:
                result = np.log1p(values, where=~np.isnan(values), out=np.full_like(values, np.nan))
                return pd.Series(result, index=X.index, name=column)
            signed = np.sign(values) * np.log1p(np.abs(values))
            return pd.Series(signed, index=X.index, name=column)
        if kind == "ratio":
            numerator = X[sources[0]].to_numpy(dtype="float64")
            denominator = X[sources[1]].to_numpy(dtype="float64")
            safe = np.where(
                (np.abs(denominator) < self.safe_division_eps) | np.isnan(denominator),
                np.nan,
                denominator,
            )
            return pd.Series(numerator / safe, index=X.index, name=key)
        if kind == "difference":
            return X[sources[0]].astype("float64") - X[sources[1]].astype("float64")
        if kind == "interaction":
            return X[sources[0]].astype("float64") * X[sources[1]].astype("float64")
        if kind == "polynomial":
            degree = int(key.rsplit("^", 1)[1])
            return X[sources[0]].astype("float64") ** degree
        raise PipelineError(f"unknown feature kind {kind!r}")  # pragma: no cover

    # sklearn API -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureEngineer:
        """Validate sources, build the catalog, and learn NaN-fill values on train."""
        X = self._check_input(X)
        self._columns_in_ = list(X.columns)
        specs = self._resolve_specs()
        if len(specs) > self.max_engineered:
            raise PipelineError(
                f"{len(specs)} engineered features exceed max_engineered={self.max_engineered}"
            )
        self._validate_sources(X, specs)

        reserved = set(self._columns_in_)
        catalog: list[CatalogEntry] = []
        computed: dict[str, pd.Series] = {}
        seen_names: set[str] = set()
        for kind, name, formula, rationale, sources, key in specs:
            if name in reserved or name in seen_names:
                raise PipelineError(
                    f"engineered feature name {name!r} collides with an existing column "
                    "or another engineered feature; set an explicit 'name' in the spec"
                )
            seen_names.add(name)
            computed[name] = self._compute(kind, sources, key, X)
            if kind == "log1p":
                values = X[sources[0]].to_numpy(dtype="float64")
                finite = values[~np.isnan(values)]
                if finite.size and float(np.nanmin(finite)) < 0.0:
                    column = sources[0]
                    formula = f"sign({column}) * log1p(|{column}|)"
                    rationale += " Signed variant applied: the train minimum is negative."
            catalog.append(
                CatalogEntry(
                    name=name,
                    kind=kind,
                    formula=formula,
                    rationale=rationale,
                    source_columns=list(sources),
                )
            )

        self.fill_values_: dict[str, float] = {}
        self.dropped_features_: dict[str, str] = {}
        kept_catalog: list[CatalogEntry] = []
        for entry in catalog:
            series = computed[entry.name].to_numpy(dtype="float64")
            series = np.where(np.isfinite(series), series, np.nan)
            finite = series[~np.isnan(series)]
            if finite.size == 0:
                self.dropped_features_[entry.name] = (
                    "engineered feature is entirely null/inf on train"
                )
                continue
            self.fill_values_[entry.name] = float(np.median(finite))
            kept_catalog.append(entry)

        self.catalog_ = kept_catalog
        self.engineered_columns_ = [entry.name for entry in kept_catalog]
        self.feature_names_out_ = self._columns_in_ + self.engineered_columns_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features, replace inf with NaN, and apply fitted fills."""
        if not hasattr(self, "catalog_"):
            raise ValueError("This FeatureEngineer instance is not fitted yet; call fit() first.")
        X = self._check_input(X)
        self._require_fitted_columns(X, self._columns_in_, "transform")
        if not self.engineered_columns_:
            return X

        specs = {entry.name: entry for entry in self.catalog_}
        engineered: dict[str, np.ndarray] = {}
        for entry in self.catalog_:
            kind = entry.kind
            sources = entry.source_columns
            if kind == "log1p":
                values = X[sources[0]].to_numpy(dtype="float64")
                if self._signed_log_column(sources[0]):
                    engineered[entry.name] = np.sign(values) * np.log1p(np.abs(values))
                else:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        engineered[entry.name] = np.log1p(values)
            elif kind == "ratio":
                numerator = X[sources[0]].to_numpy(dtype="float64")
                denominator = X[sources[1]].to_numpy(dtype="float64")
                safe = np.where(
                    (np.abs(denominator) < self.safe_division_eps) | np.isnan(denominator),
                    np.nan,
                    denominator,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    engineered[entry.name] = numerator / safe
            elif kind == "difference":
                engineered[entry.name] = X[sources[0]].to_numpy(dtype="float64") - X[
                    sources[1]
                ].to_numpy(dtype="float64")
            elif kind == "interaction":
                engineered[entry.name] = X[sources[0]].to_numpy(dtype="float64") * X[
                    sources[1]
                ].to_numpy(dtype="float64")
            else:  # polynomial
                degree = int(entry.formula.rsplit("**", 1)[1])
                engineered[entry.name] = X[sources[0]].to_numpy(dtype="float64") ** degree

        specs_names = list(specs)  # fitted order
        matrix = np.empty((len(X), len(specs_names)), dtype="float64")
        for position, name in enumerate(specs_names):
            column_values = engineered[name]
            column_values = np.where(np.isfinite(column_values), column_values, np.nan)
            nan_mask = np.isnan(column_values)
            column_values[nan_mask] = self.fill_values_[name]
            matrix[:, position] = column_values

        engineered_frame = pd.DataFrame(matrix, index=X.index, columns=self.engineered_columns_)
        out = pd.concat([X, engineered_frame], axis=1)
        return out[self.feature_names_out_]

    def _signed_log_column(self, column: str) -> bool:
        """Whether the fitted decision for ``column`` used the signed log variant."""
        entry = next(e for e in self.catalog_ if e.kind == "log1p" and e.source_columns == [column])
        return "sign(" in entry.formula
