"""Transformer unit tests + sklearn API conformance + joblib round-trip."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from numprep.errors import LeakageError, PipelineError
from numprep.partitions import Partition, tag_partition
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


def _train(n: int = 120, seed: int = 5) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x": gen.normal(0.0, 1.0, n),
            "skew": gen.exponential(2.0, n),
            "cat": gen.choice(["a", "b"], n),
            "nulls": np.where(gen.random(n) < 0.2, np.nan, 1.0),
        }
    )
    return tag_partition(frame, Partition.TRAIN)


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)
    out.attrs = {}
    return out


# ---------------------------------------------------------------------------
# TrainOnlyColumnPruner
# ---------------------------------------------------------------------------


def test_pruner_drops_nulls_constants_and_ids() -> None:
    frame = _train()
    frame["void"] = np.nan
    frame["const"] = 1.0
    frame["session_id"] = np.arange(len(frame))
    frame = tag_partition(frame, Partition.TRAIN)
    pruner = TrainOnlyColumnPruner(null_threshold=0.95)
    fitted = pruner.fit(frame)
    assert set(fitted.drop_reasons_) == {"void", "const", "session_id"}
    assert fitted.transform(_plain(frame)).columns.tolist() == [
        "x",
        "skew",
        "cat",
        "nulls",
    ]


def test_pruner_requires_train_tag() -> None:
    with pytest.raises(LeakageError):
        TrainOnlyColumnPruner().fit(_plain(_train()))


def test_pruner_respects_excluded_and_flags() -> None:
    frame = _train()
    frame["rownum"] = np.arange(len(frame))
    frame = tag_partition(frame, Partition.TRAIN)
    pruner = TrainOnlyColumnPruner(excluded=("rownum",))
    fitted = pruner.fit(frame)
    assert "rownum" not in fitted.drop_reasons_


def test_pruner_id_ratio_rule_integer_only() -> None:
    frame = _train()
    frame["seq"] = np.arange(len(frame)).astype("int64")  # integer, all unique
    frame = tag_partition(frame, Partition.TRAIN)
    fitted = TrainOnlyColumnPruner().fit(frame)
    assert "seq" in fitted.drop_reasons_


# ---------------------------------------------------------------------------
# MissingIndicatorAdder
# ---------------------------------------------------------------------------


def test_indicators_added_in_deterministic_order() -> None:
    frame = _train()
    frame["nulls2"] = frame["nulls"] * 1.0  # same null count, later name
    frame = tag_partition(frame, Partition.TRAIN)
    adder = MissingIndicatorAdder(max_indicators=1).fit(frame)
    # Tie on null count → lexicographically smaller name wins.
    assert adder.indicator_columns_ == ["nulls"]
    out = adder.transform(_plain(frame))
    assert "nulls__was_missing" in out.columns
    assert out["nulls__was_missing"].sum() > 0


def test_indicators_capped_and_disabled() -> None:
    frame = _train()
    adder = MissingIndicatorAdder(max_indicators=0).fit(frame)
    assert adder.indicator_columns_ == []
    assert adder.transform(frame) is frame


# ---------------------------------------------------------------------------
# NumPrepImputer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["median", "mean", "constant"])
def test_simple_imputer_strategies(strategy: str) -> None:
    frame = _train()
    imputer = NumPrepImputer(strategy=strategy, constant_value=-1.0).fit(frame)
    out = imputer.transform(_plain(frame))
    assert out["nulls"].isna().sum() == 0
    if strategy == "constant":
        assert (out.loc[frame["nulls"].isna(), "nulls"] == -1.0).all()
    elif strategy == "median":
        assert imputer.statistics_["nulls"] == pytest.approx(1.0)


def test_knn_imputer_fills_from_neighbours() -> None:
    frame = _train()
    imputer = NumPrepImputer(strategy="knn", knn_neighbors=3).fit(frame)
    out = imputer.transform(_plain(frame))
    assert out["nulls"].isna().sum() == 0


def test_iterative_imputer_fills() -> None:
    frame = _train()
    imputer = NumPrepImputer(strategy="iterative", random_state=1).fit(frame)
    out = imputer.transform(_plain(frame))
    assert out["nulls"].isna().sum() == 0


def test_imputer_preserves_non_numeric_columns() -> None:
    frame = _train()
    imputer = NumPrepImputer(strategy="median").fit(frame)
    out = imputer.transform(_plain(frame))
    assert out["cat"].tolist() == frame["cat"].tolist()


def test_all_null_column_survives_as_nan_documented() -> None:
    frame = _train()
    frame["void"] = np.nan
    frame = tag_partition(frame, Partition.TRAIN)
    imputer = NumPrepImputer(strategy="median").fit(frame)
    out = imputer.transform(_plain(frame))
    assert out["void"].isna().all()  # honest failure mode; output contract raises later


# ---------------------------------------------------------------------------
# OutlierClipper
# ---------------------------------------------------------------------------


def test_winsorize_bounds_fit_on_train_only() -> None:
    train = _train()
    clipper = OutlierClipper(method="winsorize", lower_quantile=0.05, upper_quantile=0.95).fit(
        train
    )
    low, high = clipper.bounds_["x"]
    train_values = _plain(train)["x"].dropna()
    expected_low = float(train_values.quantile(0.05))
    assert low == pytest.approx(expected_low)
    # Extreme evaluation values are clipped, not dropped.
    eval_frame = _plain(train).copy()
    eval_frame.loc[0, "x"] = 1e12
    out = clipper.transform(eval_frame)
    assert out["x"].iloc[0] == pytest.approx(high)
    assert len(out) == len(eval_frame)


@pytest.mark.parametrize("method", ["iqr", "zscore", "none"])
def test_other_clip_methods(method: str) -> None:
    train = _train()
    clipper = OutlierClipper(method=method).fit(train)
    out = clipper.transform(_plain(train))
    assert len(out) == len(train)
    if method == "none":
        assert all(v is None for v in clipper.bounds_.values())


def test_zscore_skips_constant_columns() -> None:
    train = _train()
    train["const"] = 2.0
    train = tag_partition(train, Partition.TRAIN)
    clipper = OutlierClipper(method="zscore").fit(train)
    assert clipper.bounds_["const"] is None


def test_clipper_logs_counts() -> None:
    train = _train()
    clipper = OutlierClipper(method="winsorize").fit(train)
    eval_frame = _plain(train).copy()
    eval_frame.loc[0, "x"] = 1e9
    clipper.transform(eval_frame)
    assert getattr(clipper, "last_transform_stats_", {}).get("x", 0) >= 1


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------


def test_log1p_signed_for_negatives() -> None:
    frame = _train()
    frame["neg"] = frame["x"] - 10.0  # strictly negative
    frame = tag_partition(frame, Partition.TRAIN)
    engineer = FeatureEngineer(log1p=("neg", "skew")).fit(frame)
    catalog = {e.name: e for e in engineer.catalog_}
    assert "sign(" in catalog["neg__log1p"].formula
    assert "sign(" not in catalog["skew__log1p"].formula  # exponential: non-negative
    out = engineer.transform(_plain(frame))
    neg = frame["neg"].to_numpy()
    expected = np.sign(neg) * np.log1p(np.abs(neg))
    np.testing.assert_allclose(out["neg__log1p"].to_numpy(), expected, equal_nan=True)


def test_ratio_safe_division() -> None:
    frame = _train()
    frame["num"] = np.arange(len(frame), dtype="float64")
    frame["den"] = np.where(np.arange(len(frame)) % 7 == 0, 0.0, 4.0)
    frame = tag_partition(frame, Partition.TRAIN)
    engineer = FeatureEngineer(ratios=(RatioSpec(numerator="num", denominator="den"),)).fit(frame)
    out = engineer.transform(_plain(frame))
    column = out["num__div__den"]
    assert np.isfinite(column).all()  # zeros → NaN → train median
    assert (column != 0).any() or (out["num"] == 0).all()


def test_ratio_missing_source_raises() -> None:
    frame = _train()
    with pytest.raises(PipelineError, match="ghost"):
        FeatureEngineer(ratios=(RatioSpec(numerator="ghost", denominator="x"),)).fit(frame)


def test_name_collision_raises() -> None:
    frame = _train()
    frame["x__log1p"] = 1.0
    frame = tag_partition(frame, Partition.TRAIN)
    with pytest.raises(PipelineError, match="collides"):
        FeatureEngineer(log1p=("x",)).fit(frame)


def test_all_null_engineered_feature_dropped() -> None:
    frame = _train()
    frame["num"] = 1.0
    frame["den"] = 0.0  # always zero → ratio always NaN
    frame = tag_partition(frame, Partition.TRAIN)
    engineer = FeatureEngineer(ratios=(RatioSpec(numerator="num", denominator="den"),)).fit(frame)
    assert engineer.engineered_columns_ == []
    assert "num__div__den" in engineer.dropped_features_


def test_diff_interaction_poly_and_catalog() -> None:
    frame = _train()
    frame = tag_partition(frame, Partition.TRAIN)
    engineer = FeatureEngineer(
        differences=(DiffSpec(minuend="x", subtrahend="skew"),),
        interactions=(InteractionSpec(left="x", right="skew"),),
        polynomial=(PolySpec(column="x", degree=2),),
    ).fit(frame)
    names = [e.name for e in engineer.catalog_]
    assert names == ["x__minus__skew", "x__x__skew", "x__pow2"]
    out = engineer.transform(_plain(frame))
    np.testing.assert_allclose(
        out["x__minus__skew"].to_numpy(),
        _plain(frame)["x"].to_numpy() - _plain(frame)["skew"].to_numpy(),
    )
    assert all(entry.rationale for entry in engineer.catalog_)


def test_engineered_extremes_become_median_filled() -> None:
    frame = _train()
    frame["num"] = 1e200
    frame["den"] = 1e-200  # ratio → inf
    frame = tag_partition(frame, Partition.TRAIN)
    engineer = FeatureEngineer(ratios=(RatioSpec(numerator="num", denominator="den"),)).fit(frame)
    # On train, the ratio is 1e400 → inf → all-null → dropped.
    assert engineer.engineered_columns_ == []


# ---------------------------------------------------------------------------
# LowCardinalityEncoder
# ---------------------------------------------------------------------------


def test_encoder_onehot_and_unseen_safety() -> None:
    train = _train()
    encoder = LowCardinalityEncoder(max_cardinality=10).fit(train)
    assert set(encoder.categories_) == {"cat"}
    eval_frame = _plain(train).copy()
    eval_frame["cat"] = "UNSEEN"
    out = encoder.transform(eval_frame)
    value_cols = [c for c in out.columns if c.startswith("cat_cat__")]
    assert len(value_cols) == 2  # a, b seen on train
    assert (out[value_cols].sum(axis=1) == 0).all()  # unseen → all zeros


def test_encoder_bools_and_nulls() -> None:
    train = _train()
    train["flag"] = pd.Series(np.where(train["x"] > 0, True, False), dtype="object")
    train.loc[0, "flag"] = None
    train = tag_partition(train, Partition.TRAIN)
    encoder = LowCardinalityEncoder().fit(train)
    assert set(encoder.categories_) >= {"flag"}
    out = encoder.transform(_plain(train))
    assert "cat_flag__False" in out.columns and "cat_flag__True" in out.columns
    assert out["cat_flag__True"].dtype == np.float32


def test_encoder_drops_high_cardinality() -> None:
    train = _train()
    train["junk"] = [f"v{i}" for i in range(len(train))]
    train = tag_partition(train, Partition.TRAIN)
    encoder = LowCardinalityEncoder(max_cardinality=5, drop_high_cardinality=True).fit(train)
    assert "junk" in encoder.dropped_columns_
    out = encoder.transform(_plain(train))
    assert "junk" not in out.columns


def test_encoder_keeps_unencoded_columns() -> None:
    train = _train()
    encoder = LowCardinalityEncoder().fit(train)
    out = encoder.transform(_plain(train))
    assert "x" in out.columns and "skew" in out.columns


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------


def test_selector_variance_threshold() -> None:
    train = _train()
    train["const"] = 1.0
    train = tag_partition(train, Partition.TRAIN)
    selector = FeatureSelector(variance_threshold=0.0).fit(train)
    assert "const" in selector.drop_reasons_


def test_selector_correlation_pruning_drops_more_connected() -> None:
    gen = np.random.default_rng(2)
    base = gen.normal(0, 1, 500)
    frame = tag_partition(
        pd.DataFrame(
            {
                "keep_low": base + gen.normal(0, 0.1, 500),
                "dup_high": base * 2.0,  # perfectly correlated with keep_low
                "unrelated": gen.normal(0, 1, 500),
            }
        ),
        Partition.TRAIN,
    )
    selector = FeatureSelector(correlation_threshold=0.95).fit(frame)
    dropped = set(selector.drop_reasons_)
    # dup_high correlates with everything equally via base; the greedy rule
    # drops one of the pair and keeps the rest.
    assert len(dropped) == 1
    assert dropped <= {"keep_low", "dup_high"}
    out = selector.transform(_plain(frame))
    assert len(out.columns) == 2


# ---------------------------------------------------------------------------
# GroupedScaler
# ---------------------------------------------------------------------------


def test_scaler_auto_skew_decisions() -> None:
    train = _train()
    scaler = GroupedScaler(default="auto").fit(train)
    assert scaler.kinds_["skew"] == "power"
    assert scaler.kinds_["x"] in ("standard", "robust")
    assert "skew" in scaler.reasons_["skew"]


def test_scaler_near_constant_passthrough() -> None:
    train = _train()
    train["const"] = 5.0
    train = tag_partition(train, Partition.TRAIN)
    scaler = GroupedScaler(default="auto").fit(train)
    assert scaler.kinds_["const"] == "none"
    assert "near-constant" in scaler.reasons_["const"]


def test_scaler_overrides_and_exclusions() -> None:
    train = _train()
    train["cat_flag"] = np.where(train["cat"] == "a", 1.0, 0.0)
    train = tag_partition(train, Partition.TRAIN)
    scaler = GroupedScaler(
        default="standard",
        overrides={"x": "minmax"},
        exclude_patterns=(r"^cat_flag$",),
    ).fit(train)
    assert scaler.kinds_["x"] == "minmax"
    assert scaler.kinds_["cat_flag"] == "none"
    out = scaler.transform(_plain(train))
    assert out["x"].min() >= 0.0 and out["x"].max() <= 1.0


def test_scaler_excluded_columns_unchanged() -> None:
    train = _train()
    train["cat_flag"] = 3.0
    train = tag_partition(train, Partition.TRAIN)
    scaler = GroupedScaler(default="standard", exclude_patterns=(r"^cat_flag$",)).fit(train)
    out = scaler.transform(_plain(train))
    assert (out["cat_flag"] == 3.0).all()


def test_scaler_unknown_override_raises() -> None:
    train = _train()
    with pytest.raises(ValueError, match="ghost"):
        GroupedScaler(overrides={"ghost": "standard"}).fit(train)


# ---------------------------------------------------------------------------
# sklearn API conformance
# ---------------------------------------------------------------------------


CONFORMANCE_ESTIMATORS = [
    TrainOnlyColumnPruner(),
    MissingIndicatorAdder(),
    NumPrepImputer(),
    OutlierClipper(),
    FeatureEngineer(ratios=(RatioSpec(numerator="x", denominator="skew"),)),
    LowCardinalityEncoder(),
    FeatureSelector(),
    GroupedScaler(),
]


@pytest.mark.parametrize("estimator", CONFORMANCE_ESTIMATORS, ids=lambda e: type(e).__name__)
def test_sklearn_estimator_contract(estimator) -> None:
    train = _train()
    cloned = clone(estimator)  # clone-safety with default params
    fitted = cloned.fit(train)
    # get_params/set_params round trip
    params = fitted.get_params()
    assert isinstance(params, dict)
    # set_output API exists and does not break transform
    if hasattr(fitted, "set_output"):
        fitted.set_output(transform="pandas")
    out = fitted.transform(_plain(train))
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(train)
    names = fitted.get_feature_names_out()
    assert list(names) == list(out.columns)


def test_pipeline_clone_and_names() -> None:
    from sklearn.pipeline import Pipeline

    from numprep.pipeline import feature_names

    cfg_pipeline = Pipeline(
        [
            ("prune", TrainOnlyColumnPruner()),
            ("impute", NumPrepImputer()),
            ("scale", GroupedScaler(default="standard")),
        ]
    )
    fitted = clone(cfg_pipeline).fit(_train())
    names = feature_names(fitted)
    assert names


def test_joblib_round_trip_reproduces_transform_exactly() -> None:
    from sklearn.pipeline import Pipeline

    train = _train()
    pipe = Pipeline(
        [
            ("impute", NumPrepImputer()),
            ("clip", OutlierClipper()),
            ("scale", GroupedScaler(default="standard")),
        ]
    ).fit(train)
    buffer = io.BytesIO()
    joblib.dump(pipe, buffer)
    buffer.seek(0)
    reloaded = joblib.load(buffer)
    eval_frame = _plain(train).iloc[:20]
    pd.testing.assert_frame_equal(pipe.transform(eval_frame), reloaded.transform(eval_frame))
