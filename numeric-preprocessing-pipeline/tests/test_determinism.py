"""Determinism tests: identical outputs across runs, processes, and hash seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conftest import run_cli
from numprep.config import load_config
from numprep.orchestrator import run_pipeline
from numprep.synthesize import generate_messy_dataset, save_messy_csv


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def prepared_csv(tmp_path: Path) -> Path:
    frame = generate_messy_dataset(rows=400, seed=5, task="regression")
    return save_messy_csv(frame, tmp_path / "raw" / "data.csv")


def _pipeline_cfg(csv: Path, out: Path):
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.output.directory = out
    cfg.target.column = "target"
    cfg.validation.input.max_null_rate = {"*": 0.5, "all_null_col": 1.0, "mostly_null": 0.99}
    return cfg


def test_same_process_reruns_are_byte_identical(tmp_path: Path, prepared_csv: Path) -> None:
    out = tmp_path / "out"
    first = run_pipeline(_pipeline_cfg(prepared_csv, out), log_level="ERROR")
    hashes_1 = dict(first["output_hashes"])
    second = run_pipeline(_pipeline_cfg(prepared_csv, out), log_level="ERROR")
    assert second["output_hashes"] == hashes_1
    for name in ("train", "val", "test"):
        assert (out / "data" / f"{name}.parquet").exists()
    # Manifest deterministic blocks equal; runtime blocks differ (timestamps).
    import json

    manifest = json.loads((out / "artifacts" / "manifest.json").read_text())
    assert manifest["deterministic"]["output_files"]["train"]["sha256"] == hashes_1["train"]


def test_fresh_processes_with_different_hash_seeds_are_identical(
    tmp_path: Path, prepared_csv: Path
) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(
        yaml.safe_dump(
            {
                "target": {"column": "target"},
                "validation": {
                    "input": {"max_null_rate": {"*": 0.5, "all_null_col": 1.0, "mostly_null": 0.99}}
                },
            }
        )
    )
    args_base = [
        "run",
        "--config",
        str(cfg_path),
        "--input",
        str(prepared_csv),
        "--seed",
        "42",
        "--log-level",
        "ERROR",
    ]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = run_cli([*args_base, "--output", str(out_a)], env_extra={"PYTHONHASHSEED": "0"})
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_cli([*args_base, "--output", str(out_b)], env_extra={"PYTHONHASHSEED": "12345"})
    assert result_b.returncode == 0, result_b.stderr

    for name in ("train", "val", "test"):
        assert _sha256(out_a / "data" / f"{name}.parquet") == _sha256(
            out_b / "data" / f"{name}.parquet"
        )

    manifest_a = json.loads((out_a / "artifacts" / "manifest.json").read_text())
    manifest_b = json.loads((out_b / "artifacts" / "manifest.json").read_text())
    assert manifest_a["deterministic"] == manifest_b["deterministic"]
    # Runtime blocks are deliberately separate:
    assert "started_at_utc" in manifest_a["runtime"]


def test_different_seeds_produce_different_splits(tmp_path: Path, prepared_csv: Path) -> None:
    out_a = tmp_path / "s1"
    out_b = tmp_path / "s2"
    run_pipeline(_pipeline_cfg(prepared_csv, out_a), log_level="ERROR")
    cfg_b = _pipeline_cfg(prepared_csv, out_b)
    cfg_b.run.seed = 43
    run_pipeline(cfg_b, log_level="ERROR")
    assert _sha256(out_a / "data" / "train.parquet") != _sha256(out_b / "data" / "train.parquet")


def test_no_timestamps_inside_data_files(tmp_path: Path, prepared_csv: Path) -> None:
    out = tmp_path / "out"
    run_pipeline(_pipeline_cfg(prepared_csv, out), log_level="ERROR")
    import polars as pl

    for name in ("train", "val", "test"):
        schema = pl.read_parquet_schema(out / "data" / f"{name}.parquet")
        assert all("datetime" not in str(dtype) for dtype in schema.values())
