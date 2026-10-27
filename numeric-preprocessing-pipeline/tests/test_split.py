"""Split tests: sizing, strategies, fallbacks, overlap invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numprep.config import load_config
from numprep.errors import PipelineError
from numprep.ingest import ROW_INDEX_COL
from numprep.partitions import Partition, partition_of
from numprep.split import _largest_remainder_sizes, detect_task, split_data


def _frame(n: int = 300, seed: int = 3, rare_class: bool = False) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    target = gen.integers(0, 2, n).astype("int64")
    if rare_class:
        target[:2] = 7  # a class with only 2 members (< partitions)
    return pd.DataFrame(
        {
            "x": gen.normal(0, 1, n),
            "grp": np.array([f"g{i % 23}" for i in range(n)]),
            "target": target,
            ROW_INDEX_COL: np.arange(n, dtype="uint32"),
        }
    )


def test_largest_remainder_exact_sum() -> None:
    sizes = _largest_remainder_sizes(1000, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert sum(sizes.values()) == 1000
    assert all(size >= 1 for size in sizes.values())
    sizes = _largest_remainder_sizes(7, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert sum(sizes.values()) == 7 and all(size >= 1 for size in sizes.values())


def test_split_sizes_match_ratios() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(_frame(), cfg)
    n = 300
    assert result.meta.sizes["train"] == 210
    assert result.meta.sizes["val"] + result.meta.sizes["test"] == 90
    assert sum(result.meta.sizes.values()) == n


def test_partitions_are_tagged_and_disjoint() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(_frame(), cfg)
    assert partition_of(result.train) is Partition.TRAIN
    assert partition_of(result.val) is Partition.VAL
    assert partition_of(result.test) is Partition.TEST
    ids = {k: set(result.partitions()[k][ROW_INDEX_COL]) for k in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"])
    assert not (ids["train"] & ids["test"])
    assert not (ids["val"] & ids["test"])


def test_stratified_proportions_preserved() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "stratified"
    result = split_data(_frame(), cfg)

    def proportion(frame: pd.DataFrame, value: int) -> float:
        return float((frame["target"] == value).mean())

    train_p = proportion(result.train, 1)
    val_p = proportion(result.val, 1)
    test_p = proportion(result.test, 1)
    assert abs(train_p - val_p) < 0.05
    assert abs(train_p - test_p) < 0.05
    assert result.meta.fallback is None


def test_stratified_falls_back_for_rare_class() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "stratified"
    result = split_data(_frame(rare_class=True), cfg)
    assert result.meta.fallback is not None
    assert "rare class" in result.meta.fallback
    assert any("stratified" in w for w in result.meta.warnings)
    # The fallback still produces a valid, disjoint split.
    assert sum(result.meta.sizes.values()) == 300


def test_group_split_zero_overlap() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "group"
    cfg.split.group_columns = ["grp"]
    result = split_data(_frame(), cfg)
    assert result.meta.group_overlap_zero is True

    def groups(frame: pd.DataFrame) -> set[str]:
        return set(frame["grp"])

    g_train, g_val, g_test = groups(result.train), groups(result.val), groups(result.test)
    assert not (g_train & g_val or g_train & g_test or g_val & g_test)


def test_group_split_fallback_when_too_few_groups() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "group"
    cfg.split.group_columns = ["grp"]
    frame = _frame()
    frame["grp"] = "only"  # a single group
    result = split_data(frame, cfg)
    assert result.meta.fallback is not None
    assert "group(s)" in result.meta.fallback


def test_chronological_is_ordered_and_nonoverlapping() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "chronological"
    cfg.split.time_column = "ts"
    frame = _frame()
    frame["ts"] = pd.date_range("2024-01-01", periods=300, freq="min")
    result = split_data(frame, cfg)
    train_max = result.train["ts"].max()
    val_min = result.val["ts"].min()
    val_max = result.val["ts"].max()
    test_min = result.test["ts"].min()
    assert train_max < val_min
    assert val_max < test_min
    assert result.meta.time_ranges is not None
    assert result.meta.time_ranges["train"]["max"] is not None


def test_chronological_drops_null_times_with_warning() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "chronological"
    cfg.split.time_column = "ts"
    frame = _frame()
    frame["ts"] = pd.date_range("2024-01-01", periods=300, freq="min").astype(str)
    frame.loc[[3, 10, 20], "ts"] = None
    result = split_data(frame, cfg)
    assert any("3 row(s)" in w for w in result.meta.warnings)
    assert sum(result.meta.sizes.values()) == 297


def test_ratios_renormalized_with_warning() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.ratios = {"train": 0.7, "val": 0.2, "test": 0.2}  # sums to 1.1
    result = split_data(_frame(), cfg)
    assert result.meta.ratios_normalized is True
    assert abs(sum(result.meta.applied_ratios.values()) - 1.0) < 1e-9
    assert any("renormalized" in w for w in result.meta.warnings)


def test_too_few_rows_raise() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    with pytest.raises(PipelineError, match="at least 3"):
        split_data(_frame(n=2), cfg)


def test_minimum_viable_split_three_rows() -> None:
    cfg = load_config()
    cfg.target.column = "target"
    result = split_data(_frame(n=3), cfg)
    assert result.meta.sizes == {"train": 1, "val": 1, "test": 1}


def test_missing_group_column_raises_contract_error() -> None:
    from numprep.errors import InputContractError

    cfg = load_config()
    cfg.target.column = "target"
    cfg.split.strategy = "group"
    cfg.split.group_columns = ["ghost"]
    with pytest.raises(InputContractError, match="ghost"):
        split_data(_frame(), cfg)


def test_detect_task_heuristics() -> None:
    assert detect_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert detect_task(pd.Series([True, False])) == "classification"
    assert detect_task(pd.Series(["a", "b"])) == "classification"
    assert detect_task(pd.Series([0.5, 1.5, -2.3])) == "regression"
    # integral floats with few uniques → classification
    assert detect_task(pd.Series([0.0, 1.0, 1.0, 0.0])) == "classification"
    # integral floats with many uniques → regression
    assert detect_task(pd.Series(np.arange(50, dtype="float64"))) == "regression"
