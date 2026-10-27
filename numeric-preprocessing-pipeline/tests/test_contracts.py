"""Contract tests: input violations exit 3, output violations exit 4."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from numprep.config import RangeSpec, load_config
from numprep.contracts import validate_input_contract, validate_output_contract
from numprep.errors import InputContractError, OutputContractError
from numprep.ingest import ROW_INDEX_COL


def _cfg(**overrides):
    cfg = load_config()
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(cfg, section), field, value)
    return cfg


def test_missing_required_column_aggregates() -> None:
    cfg = _cfg()
    cfg.validation.input.required_columns = ["ghost_a", "ghost_b"]
    frame = pl_from_pandas(pd.DataFrame({"a": [1.0]}))
    with pytest.raises(InputContractError) as excinfo:
        validate_input_contract(frame, cfg)
    message = str(excinfo.value)
    assert "ghost_a" in message and "ghost_b" in message  # all violations at once


def pl_from_pandas(frame: pd.DataFrame):
    import polars as pl

    return pl.from_pandas(frame).with_row_index(ROW_INDEX_COL)


def test_null_rate_ceiling_violation_lists_column() -> None:
    cfg = _cfg()
    cfg.validation.input.max_null_rate = {"a": 0.1}
    frame = pl_from_pandas(pd.DataFrame({"a": [1.0, None, None, None]}))
    with pytest.raises(InputContractError, match=r"null rate 0\.75"):
        validate_input_contract(frame, cfg)


def test_default_ceiling_applies_via_star() -> None:
    cfg = _cfg()
    cfg.validation.input.max_null_rate = {"*": 0.1}
    frame = pl_from_pandas(pd.DataFrame({"a": [1.0, None], "b": [1.0, 2.0]}))
    with pytest.raises(InputContractError, match=r"'a' null rate 0\.5"):
        validate_input_contract(frame, cfg)


def test_range_violations_via_pandera() -> None:
    cfg = _cfg()
    cfg.validation.input.ranges = {"a": RangeSpec(min=0.0, max=10.0)}
    frame = pl_from_pandas(pd.DataFrame({"a": [5.0, 50.0]}))
    with pytest.raises(InputContractError, match="range check"):
        validate_input_contract(frame, cfg)


def test_expected_numeric_flags_string_column() -> None:
    cfg = _cfg()
    cfg.validation.input.expected_numeric = ["a"]
    frame = pl.from_pandas(pd.DataFrame({"a": ["x", "y"]})).with_row_index(ROW_INDEX_COL)
    with pytest.raises(InputContractError, match="expected numeric"):
        validate_input_contract(frame, cfg)


def test_disabled_contract_is_noop() -> None:
    cfg = _cfg()
    cfg.validation.input.enabled = False
    frame = pl_from_pandas(pd.DataFrame({"a": [None, None]}))
    summary = validate_input_contract(frame, cfg)
    assert summary["enabled"] is False


def test_output_contract_nan_inf_detected() -> None:
    cfg = _cfg()
    frame = pd.DataFrame({"a": [1.0, np.inf], "b": [1.0, np.nan]})
    with pytest.raises(OutputContractError) as excinfo:
        validate_output_contract(
            {"train": frame, "val": frame.copy(), "test": frame.copy()}, None, cfg
        )
    assert "inf" in str(excinfo.value) and "NaN" in str(excinfo.value)


def test_output_contract_schema_drift_detected() -> None:
    cfg = _cfg()
    train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    val = pd.DataFrame({"a": [1.0], "b_int": [2.0]})
    test = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(OutputContractError, match="column order differs"):
        validate_output_contract({"train": train, "val": val, "test": test}, None, cfg)


def test_output_contract_dtype_drift_detected() -> None:
    cfg = _cfg()
    train = pd.DataFrame({"a": [1.0], "b": [2.0]})
    val = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(OutputContractError, match="dtype"):
        validate_output_contract({"train": train, "val": val, "test": train.copy()}, None, cfg)


def test_output_contract_empty_split_detected() -> None:
    cfg = _cfg()
    train = pd.DataFrame({"a": [1.0]})
    with pytest.raises(OutputContractError, match="empty"):
        validate_output_contract(
            {"train": train, "val": train.copy(), "test": train.iloc[:0]}, None, cfg
        )


def test_output_contract_missing_target_detected() -> None:
    cfg = _cfg()
    train = pd.DataFrame({"a": [1.0]})
    with pytest.raises(OutputContractError, match="lacks target"):
        validate_output_contract(
            {"train": train, "val": train.copy(), "test": train.copy()}, "target", cfg
        )


def test_output_contract_non_numeric_feature_detected() -> None:
    cfg = _cfg()
    frame = pd.DataFrame({"a": [1.0], "text": ["x"]})
    with pytest.raises(OutputContractError, match="non-numeric"):
        validate_output_contract(
            {"train": frame, "val": frame.copy(), "test": frame.copy()}, None, cfg
        )


def test_output_contract_happy_path_summary() -> None:
    cfg = _cfg()
    frame = pd.DataFrame({"a": [1.0, 2.0], "target": [0.5, 0.7], ROW_INDEX_COL: [0, 1]})
    summary = validate_output_contract(
        {"train": frame, "val": frame.copy(), "test": frame.copy()}, "target", cfg
    )
    assert summary["partitions"] == {"train": 2, "val": 2, "test": 2}
    assert summary["columns"] == 3
