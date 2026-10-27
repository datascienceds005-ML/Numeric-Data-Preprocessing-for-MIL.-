"""Leakage tests — the critical suite (DESIGN §4, brief §5).

Every subtle leakage source named in the brief has a dedicated test here.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from numprep.config import load_config
from numprep.errors import LeakageError
from numprep.ingest import ROW_INDEX_COL
from numprep.partitions import Partition, partition_of, require_train, tag_partition
from numprep.pipeline import build_pipeline, feature_matrix, fit_preprocessor, transform_partition
from numprep.split import split_data
from numprep.transformers.impute import NumPrepImputer
from numprep.transformers.outliers import OutlierClipper
from numprep.transformers.prune import TrainOnlyColumnPruner


def _leak_frame(n: int = 400, seed: int = 17) -> pd.DataFrame:
    """Train and evaluation rows come from *different* distributions, so any
    statistic fitted on the union is visibly different from train-only."""
    gen = np.random.default_rng(seed)
    train_values = gen.normal(100.0, 2.0, n)
    eval_values = gen.normal(200.0, 2.0, 2 * n)  # deliberately different
    frame = pd.DataFrame(
        {
            "x": np.concatenate([train_values, eval_values]),
            "target": gen.normal(0, 1, 3 * n),
            ROW_INDEX_COL: np.arange(3 * n, dtype="uint32"),
        }
    )
    return frame


@pytest.fixture()
def partitions() -> dict[str, pd.DataFrame]:
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(_leak_frame(), cfg)
    return result.partitions()


def test_fit_rejects_val_test_and_unsplit(partitions: dict[str, pd.DataFrame]) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    pipe = build_pipeline(cfg)
    for name in ("val", "test"):
        with pytest.raises(LeakageError):
            fit_preprocessor(pipe, partitions[name], cfg)
    with pytest.raises(LeakageError):
        fit_preprocessor(pipe, pd.concat(list(partitions.values())), cfg)
    # And the untagged full frame:
    unsplit = _leak_frame()
    with pytest.raises(LeakageError):
        fit_preprocessor(pipe, unsplit, cfg)


def test_require_train_guard_direct() -> None:
    with pytest.raises(LeakageError, match="untagged"):
        require_train(pd.DataFrame({"a": [1]}))
    val = tag_partition(pd.DataFrame({"a": [1]}), Partition.VAL)
    with pytest.raises(LeakageError, match="VAL"):
        require_train(val)
    train = tag_partition(pd.DataFrame({"a": [1]}), Partition.TRAIN)
    assert require_train(train) is train


def test_imputer_statistics_equal_train_only() -> None:
    """Partitions with genuinely different distributions: fitting on the union
    would give ~166; fitting on train alone must give ~100."""
    gen = np.random.default_rng(17)
    train = tag_partition(pd.DataFrame({"x": gen.normal(100.0, 2.0, 300)}), Partition.TRAIN)
    val = tag_partition(pd.DataFrame({"x": gen.normal(200.0, 2.0, 300)}), Partition.VAL)
    test = tag_partition(pd.DataFrame({"x": gen.normal(200.0, 2.0, 300)}), Partition.TEST)
    imputer = NumPrepImputer(strategy="median").fit(train)
    assert imputer.statistics_["x"] == pytest.approx(float(train["x"].median()), abs=1e-9)
    full_median = float(pd.concat([train, val, test])["x"].median())
    assert abs(imputer.statistics_["x"] - full_median) > 50.0


def test_clip_bounds_equal_train_quantiles(partitions: dict[str, pd.DataFrame]) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    train = partitions["train"]
    clipper = OutlierClipper(method="winsorize", lower_quantile=0.1, upper_quantile=0.9).fit(
        feature_matrix(train, cfg)
    )
    low, high = clipper.bounds_["x"]
    train_x = feature_matrix(train, cfg)["x"]
    assert low == pytest.approx(float(train_x.quantile(0.1)))
    assert high == pytest.approx(float(train_x.quantile(0.9)))


def test_corrupting_eval_partitions_leaves_artifact_byte_identical(
    partitions: dict[str, pd.DataFrame],
) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    train = partitions["train"]
    pipe_a = build_pipeline(cfg)
    fit_preprocessor(pipe_a, train, cfg)
    buffer_a = io.BytesIO()
    joblib.dump(pipe_a, buffer_a)

    corrupted = {name: frame.copy() for name, frame in partitions.items() if name != "train"}
    for frame in corrupted.values():
        for column in frame.columns:
            frame[column] = -12345.0

    pipe_b = build_pipeline(cfg)
    fit_preprocessor(pipe_b, train, cfg)  # fit path never touches val/test
    buffer_b = io.BytesIO()
    joblib.dump(pipe_b, buffer_b)
    assert buffer_a.getvalue() == buffer_b.getvalue()


def test_column_drop_decisions_fitted_on_train_only() -> None:
    """A column that is 96% null overall but only 10% null on train MUST survive."""
    gen = np.random.default_rng(23)
    n_train, n_eval = 500, 3000
    train_flag = np.where(gen.random(n_train) < 0.10, np.nan, 1.0)
    eval_flag = np.where(gen.random(n_eval) < 0.99, np.nan, 1.0)
    frame = pd.DataFrame(
        {
            "sparse_col": np.concatenate([train_flag, eval_flag]),
            "x": gen.normal(0, 1, n_train + n_eval),
            "target": gen.normal(0, 1, n_train + n_eval),
            ROW_INDEX_COL: np.arange(n_train + n_eval, dtype="uint32"),
        }
    )
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(frame, cfg)
    pruner = TrainOnlyColumnPruner(null_threshold=0.95).fit(feature_matrix(result.train, cfg))
    assert "sparse_col" not in pruner.drop_reasons_  # train null rate is 10%
    overall_null_rate = float(frame["sparse_col"].isna().mean())
    assert overall_null_rate > 0.7  # ...even though the union looks droppable


def test_group_overlap_is_zero() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "group"
    cfg.split.group_columns = ["entity"]
    gen = np.random.default_rng(31)
    frame = pd.DataFrame(
        {
            "x": gen.normal(0, 1, 600),
            "entity": np.array([f"e{i % 40}" for i in range(600)]),
            "target": gen.normal(0, 1, 600),
            ROW_INDEX_COL: np.arange(600, dtype="uint32"),
        }
    )
    result = split_data(frame, cfg)
    assert result.meta.group_overlap_zero is True
    keys = {name: set(part["entity"]) for name, part in result.partitions().items()}
    assert not (keys["train"] & keys["val"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["val"] & keys["test"])


def test_temporal_features_use_only_past() -> None:
    import polars as pl

    from numprep.temporal import add_temporal_features

    cfg = load_config()
    cfg.split.strategy = "chronological"
    cfg.split.time_column = "ts"
    cfg.cleaning.temporal.enabled = True
    cfg.cleaning.temporal.lags = [1]
    cfg.cleaning.temporal.rolling_windows = [3]
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=50, freq="min"),
            "value": np.arange(50, dtype="float64"),
            ROW_INDEX_COL: np.arange(50, dtype="uint32"),
        }
    )
    out, _ = add_temporal_features(pl.from_pandas(frame), cfg)
    values = out["value"].to_list()
    lags = out["value__lag1"].to_list()
    for t in range(1, 50):
        assert lags[t] == values[t - 1]  # strictly past
    assert lags[0] is None
    roll = out["value__rollmean3"].to_list()
    for t in range(3, 50):
        assert roll[t] == pytest.approx(np.mean(values[t - 3 : t]))  # window ends at t-1


def test_transform_preserves_row_count(partitions: dict[str, pd.DataFrame]) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    pipe = build_pipeline(cfg)
    fit_preprocessor(pipe, partitions["train"], cfg)
    for name, part in partitions.items():
        out = transform_partition(pipe, part, cfg, name)
        assert len(out) == len(part)


def test_audit_decisions_do_not_drive_fitting() -> None:
    """The pruner's decisions equal train-recomputed facts, not audit output."""
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(_leak_frame(), cfg)
    train_features = feature_matrix(result.train, cfg)
    pruner = TrainOnlyColumnPruner().fit(train_features)
    # Recompute by hand from train only:
    null_rate = float(train_features["x"].isna().mean())
    assert ("x" in pruner.drop_reasons_) == (null_rate > pruner.null_threshold)
    # The fit API's only data parameter is the train partition:
    import inspect

    params = list(inspect.signature(fit_preprocessor).parameters)
    assert params == ["pipe", "train", "cfg"]


def test_target_never_enters_the_pipeline(partitions: dict[str, pd.DataFrame]) -> None:
    cfg = load_config()
    cfg.target.column = "target"
    features = feature_matrix(partitions["train"], cfg)
    assert "target" not in features.columns
    pipe = build_pipeline(cfg)
    fitted = fit_preprocessor(pipe, partitions["train"], cfg)
    assert "target" not in fitted.named_steps["prune"]._columns_in_


def test_partition_tags_immutable_via_public_api() -> None:
    frame = tag_partition(pd.DataFrame({"a": [1]}), Partition.TRAIN)
    assert partition_of(frame) is Partition.TRAIN
    # A frame with a bogus tag value is treated as untagged:
    bogus = pd.DataFrame({"a": [1]})
    bogus.attrs["numprep.partition"] = "everything"
    assert partition_of(bogus) is None
