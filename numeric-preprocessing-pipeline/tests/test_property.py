"""Property-based tests (Hypothesis, derandomized for CI determinism)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from numprep.partitions import Partition, tag_partition
from numprep.transformers.features import FeatureEngineer, RatioSpec
from numprep.transformers.outliers import OutlierClipper
from numprep.transformers.scale import GroupedScaler

SETTINGS = settings(derandomize=True, max_examples=40, deadline=None)

# Reusable strategies: finite floats (plus NaN allowed selectively) and small frames.
finite_series = st.lists(
    st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.none(),
    ),
    min_size=8,
    max_size=64,
)


def _frame_from(values: list[float | None], name: str = "x") -> pd.DataFrame:
    return pd.DataFrame({name: np.array(values, dtype="float64")})


@SETTINGS
@given(finite_series)
def test_winsorizer_output_within_fitted_bounds(values: list[float | None]) -> None:
    frame = tag_partition(_frame_from(values), Partition.TRAIN)
    if frame["x"].dropna().empty:
        return
    clipper = OutlierClipper(method="winsorize", lower_quantile=0.1, upper_quantile=0.9).fit(frame)
    low, high = clipper.bounds_["x"]
    out = clipper.transform(_untagged(frame))
    finite = out["x"].dropna()
    assert (finite >= low - 1e-9).all()
    assert (finite <= high + 1e-9).all()
    assert len(out) == len(frame)  # never drops rows


@SETTINGS
@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), min_size=8, max_size=48),
    st.lists(st.floats(min_value=1e-6, max_value=1e6, allow_nan=False), min_size=8, max_size=48),
)
def test_safe_division_finite_when_denominator_absent_zero(
    numerators: list[float], denominators: list[float]
) -> None:
    n = min(len(numerators), len(denominators))
    frame = tag_partition(
        pd.DataFrame({"num": numerators[:n], "den": denominators[:n]}),
        Partition.TRAIN,
    )
    engineer = FeatureEngineer(ratios=(RatioSpec(numerator="num", denominator="den"),)).fit(frame)
    out = engineer.transform(_untagged(frame))
    column = out["num__div__den"].to_numpy(dtype="float64")
    assert np.isfinite(column).all()  # train medians fill any NaN


@SETTINGS
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=4, max_size=48))
def test_signed_log_finite_for_all_finite_inputs(values: list[float]) -> None:
    frame = tag_partition(_frame_from(values, name="v"), Partition.TRAIN)
    engineer = FeatureEngineer(log1p=("v",)).fit(frame)
    out = engineer.transform(_untagged(frame))
    column = out["v__log1p"].to_numpy(dtype="float64")
    assert np.isfinite(column).all()
    if min(values) < 0:
        entry = next(e for e in engineer.catalog_ if e.kind == "log1p")
        assert "sign(" in entry.formula


@SETTINGS
@given(finite_series)
def test_standard_scaler_zero_mean_unit_variance(values: list[float | None]) -> None:
    frame = tag_partition(_frame_from(values), Partition.TRAIN)
    finite = frame["x"].dropna()
    if finite.nunique() <= 1 or len(finite) < 3 or float(finite.std()) < 1e-12:
        return  # near-constant columns are passed through by design
    scaler = GroupedScaler(default="standard").fit(frame)
    assert scaler.kinds_["x"] == "standard"
    out = scaler.transform(_untagged(frame))
    transformed = out["x"].to_numpy(dtype="float64")
    assert abs(float(np.nanmean(transformed))) < 1e-9
    assert abs(float(np.nanstd(transformed)) - 1.0) < 1e-6


@SETTINGS
@given(finite_series)
def test_minmax_scaler_bounded(values: list[float | None]) -> None:
    frame = tag_partition(_frame_from(values), Partition.TRAIN)
    if frame["x"].dropna().nunique() <= 1:
        return
    scaler = GroupedScaler(default="minmax").fit(frame)
    out = scaler.transform(_untagged(frame))
    finite = out["x"].dropna()
    assert finite.min() >= -1e-9
    assert finite.max() <= 1.0 + 1e-9


def _untagged(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy(deep=False)
    out.attrs = {}
    return out
