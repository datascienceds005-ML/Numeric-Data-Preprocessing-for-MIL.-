"""Pipeline composition and the leakage-guarded fit/transform API.

``build_pipeline`` turns validated config into a single
:class:`sklearn.pipeline.Pipeline` of DataFrame-native transformers (see
DESIGN §5 for the step order and the ColumnTransformer tradeoff).

``fit_preprocessor`` accepts **only the tagged train partition**
(:func:`numprep.partitions.require_train` raises :class:`LeakageError`
otherwise) — fitting on unsplit data or on val/test is a loud runtime error,
not a silent statistic leak. ``transform_partition`` accepts any partition,
asserts it is tagged, and enforces row-count preservation.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import cast

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from numprep.config import NumPrepConfig
from numprep.errors import LeakageError, PipelineError
from numprep.ingest import ROW_INDEX_COL
from numprep.partitions import PARTITION_KEY, partition_of, require_train
from numprep.transformers.encode import LowCardinalityEncoder
from numprep.transformers.features import (
    DiffSpec,
    FeatureEngineer,
    InteractionSpec,
    PolySpec,
    RatioSpec,
)
from numprep.transformers.impute import NumPrepImputer
from numprep.transformers.indicators import MissingIndicatorAdder
from numprep.transformers.outliers import OutlierClipper
from numprep.transformers.prune import TrainOnlyColumnPruner
from numprep.transformers.scale import GroupedScaler
from numprep.transformers.select import FeatureSelector

logger = logging.getLogger("numprep")


def build_pipeline(cfg: NumPrepConfig) -> Pipeline:
    """Create the unfitted preprocessing pipeline from validated config."""
    pre = cfg.preprocessing
    features = pre.features
    steps: list[tuple[str, object]] = [
        (
            "prune",
            TrainOnlyColumnPruner(
                null_threshold=cfg.cleaning.null_column_threshold,
                drop_constant=cfg.cleaning.drop_constant_columns,
                drop_id_like=cfg.cleaning.drop_id_like,
                id_name_pattern=cfg.cleaning.id_name_pattern,
                id_unique_ratio=cfg.cleaning.id_unique_ratio,
            ),
        ),
        (
            "indicators",
            MissingIndicatorAdder(
                max_indicators=pre.imputation.max_indicators
                if pre.imputation.add_missing_indicators
                else 0,
            ),
        ),
        (
            "impute",
            NumPrepImputer(
                strategy=pre.imputation.strategy,
                constant_value=pre.imputation.constant_value,
                knn_neighbors=pre.imputation.knn_neighbors,
                iterative_max_iter=pre.imputation.iterative_max_iter,
                random_state=cfg.run.seed,
            ),
        ),
        (
            "clip",
            OutlierClipper(
                method=pre.outliers.method,
                lower_quantile=pre.outliers.lower_quantile,
                upper_quantile=pre.outliers.upper_quantile,
                iqr_k=pre.outliers.iqr_k,
                zscore_cap=pre.outliers.zscore_cap,
            ),
        ),
        (
            "features",
            FeatureEngineer(
                log1p=tuple(features.log1p),
                ratios=tuple(
                    RatioSpec(numerator=s.numerator, denominator=s.denominator, name=s.name)
                    for s in features.ratios
                ),
                differences=tuple(
                    DiffSpec(minuend=s.minuend, subtrahend=s.subtrahend, name=s.name)
                    for s in features.differences
                ),
                interactions=tuple(
                    InteractionSpec(left=s.left, right=s.right, name=s.name)
                    for s in features.interactions
                ),
                polynomial=tuple(
                    PolySpec(column=s.column, degree=s.degree, name=s.name)
                    for s in features.polynomial
                ),
                safe_division_eps=features.safe_division_eps,
                max_engineered=features.max_engineered,
            ),
        ),
        (
            "encode",
            LowCardinalityEncoder(
                max_cardinality=pre.encoding.max_cardinality,
                onehot_max_categories=pre.encoding.onehot_max_categories,
                drop_high_cardinality=pre.encoding.drop_high_cardinality,
            ),
        ),
        (
            "select",
            FeatureSelector(
                variance_threshold=pre.selection.variance_threshold,
                correlation_threshold=pre.selection.correlation_threshold,
                sample_size=pre.selection.sample_size,
            ),
        ),
        (
            "scale",
            GroupedScaler(
                default=pre.scaling.default,
                overrides=dict(pre.scaling.overrides),
                exclude_patterns=tuple(pre.scaling.exclude_patterns),
                skew_high=pre.scaling.skew_high,
                skew_moderate=pre.scaling.skew_moderate,
                quantile_output_distribution=pre.scaling.quantile_output_distribution,
                quantile_n_quantiles=pre.scaling.quantile_n_quantiles,
            ),
        ),
    ]
    return Pipeline(steps)


def feature_matrix(frame: pd.DataFrame, cfg: NumPrepConfig) -> pd.DataFrame:
    """Extract the feature matrix (target + provenance removed, tag preserved).

    The target, ``__row_index__``, and the split's time-axis column are all
    excluded: the first must never be fitted on, the second is provenance, and
    the third is ordering metadata (recoverable via ``__row_index__``), not a
    model feature.
    """
    drop_columns = [
        name
        for name in (cfg.target.column, ROW_INDEX_COL, cfg.split.time_column)
        if name is not None
    ]
    missing = [name for name in drop_columns if name not in frame.columns]
    if cfg.target.column is not None and cfg.target.column in missing:
        raise PipelineError(
            f"target column {cfg.target.column!r} is absent; cannot build the feature matrix"
        )
    features = frame.drop(columns=[c for c in drop_columns if c in frame.columns])
    role = partition_of(frame)
    if role is None:
        raise LeakageError(
            "feature_matrix() received an untagged frame; partitions must come from split_data()"
        )
    # pandas drop() propagates attrs in most versions; re-tag explicitly so the
    # train-only guard never depends on that behavior.
    features.attrs = {**features.attrs, PARTITION_KEY: role.value}
    if cfg.target.column is not None and cfg.target.column in features.columns:
        raise PipelineError("target survived feature_matrix(); refusing to fit on it")
    return features


def fit_preprocessor(pipe: Pipeline, train: pd.DataFrame, cfg: NumPrepConfig) -> Pipeline:
    """Fit the pipeline on the train partition **only**.

    Raises :class:`LeakageError` for any frame that is not the tagged train
    partition — this is the API surface that makes "fit on train only" the
    path of least resistance.
    """
    require_train(train)
    features = feature_matrix(train, cfg)
    pipe.fit(features)
    logger.info(
        "preprocessor_fitted",
        extra={
            "event": "preprocessor_fitted",
            "rows": len(features),
            "columns_in": int(features.shape[1]),
            "columns_out": len(feature_names(pipe)),
        },
    )
    return pipe


def transform_partition(
    pipe: Pipeline, partition: pd.DataFrame, cfg: NumPrepConfig, name: str
) -> pd.DataFrame:
    """Transform any tagged partition with the fitted pipeline (never refits)."""
    if partition_of(partition) is None:
        raise LeakageError(
            f"transform_partition({name!r}) received an untagged frame; "
            "partitions must come from split_data()"
        )
    features = feature_matrix(partition, cfg)
    transformed = cast(pd.DataFrame, pipe.transform(features))
    if len(transformed) != len(features):
        raise PipelineError(
            f"transforming {name} changed the row count ({len(features)} -> {len(transformed)})"
        )
    return transformed


def feature_names(pipe: Pipeline) -> list[str]:
    """Output feature names of the fitted pipeline (via get_feature_names_out)."""
    return [str(name) for name in pipe.get_feature_names_out()]


def save_preprocessor(pipe: Pipeline, path: Path) -> None:
    """Serialize the fitted pipeline atomically with joblib (compress=3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".joblib.tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(pipe, tmp_path, compress=3)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def load_preprocessor(path: Path) -> Pipeline:
    """Load a fitted pipeline saved by :func:`save_preprocessor`."""
    if not path.is_file():
        raise PipelineError(f"preprocessor artifact not found: {path}")
    return joblib.load(path)
