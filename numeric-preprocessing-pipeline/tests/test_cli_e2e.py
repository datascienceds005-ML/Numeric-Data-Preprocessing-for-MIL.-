"""End-to-end CLI tests: true subprocess exit codes, artifacts, round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import polars as pl
import pytest
import yaml

from conftest import run_cli


@pytest.fixture()
def project_csv(tmp_path: Path) -> Path:
    from numprep.synthesize import generate_messy_dataset, save_messy_csv

    return save_messy_csv(
        generate_messy_dataset(400, seed=9, task="regression"), tmp_path / "raw" / "d.csv"
    )


@pytest.fixture()
def good_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "target": {"column": "target", "task": "regression"},
                "validation": {
                    "input": {"max_null_rate": {"*": 0.5, "all_null_col": 1.0, "mostly_null": 0.99}}
                },
            }
        )
    )
    return path


def test_cli_run_happy_path(tmp_path: Path, project_csv: Path, good_cfg: Path) -> None:
    out = tmp_path / "out"
    result = run_cli(
        [
            "run",
            "--config",
            str(good_cfg),
            "--input",
            str(project_csv),
            "--output",
            str(out),
            "--seed",
            "42",
        ]
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for relpath in (
        "data/train.parquet",
        "data/val.parquet",
        "data/test.parquet",
        "artifacts/preprocessor.joblib",
        "artifacts/fitted_stats.json",
        "artifacts/feature_catalog.json",
        "artifacts/manifest.json",
        "audit/audit.json",
        "audit/audit.md",
        "reports/summary.md",
        "logs/run.jsonl",
        "snapshot/data.parquet",
    ):
        assert (out / relpath).exists(), relpath


def test_cli_round_trip_reloaded_preprocessor_matches_stored_output(
    tmp_path: Path, project_csv: Path, good_cfg: Path
) -> None:
    out = tmp_path / "out"
    result = run_cli(
        [
            "run",
            "--config",
            str(good_cfg),
            "--input",
            str(project_csv),
            "--output",
            str(out),
            "--seed",
            "42",
        ]
    )
    assert result.returncode == 0, result.stderr[-2000:]

    pipe = joblib.load(out / "artifacts" / "preprocessor.joblib")
    stored = pl.read_parquet(out / "data" / "test.parquet").to_pandas()

    # Rebuild the ORIGINAL test partition in-process (deterministic pipeline):
    # ingest → clean → split with the same seed must reproduce it exactly.
    from numprep import clean as clean_mod
    from numprep import ingest as ingest_mod
    from numprep import split as split_mod
    from numprep.config import load_config
    from numprep.jsonlog import setup_logging
    from numprep.pipeline import feature_matrix
    from numprep.seed import set_global_seed

    cfg = load_config(good_cfg)
    cfg.input.paths = [str(project_csv)]
    cfg.output.directory = out
    set_global_seed(cfg.run.seed)
    ingest_result = ingest_mod.ingest(cfg, out, setup_logging("ERROR"))
    cleaned, _ = clean_mod.clean_stateless(ingest_result.frame, cfg)
    split_result = split_mod.split_data(cleaned.to_pandas(), cfg)
    original_test = split_result.test

    recomputed = pipe.transform(feature_matrix(original_test, cfg))
    stored_features = stored.drop(columns=["target", "__row_index__"])
    pd.testing.assert_frame_equal(
        recomputed.reset_index(drop=True),
        stored_features.reset_index(drop=True),
        check_exact=True,
    )


def test_cli_exit_code_3_on_contract_violation(tmp_path: Path, project_csv: Path) -> None:
    bad_cfg = tmp_path / "bad_contract.yaml"
    bad_cfg.write_text(
        yaml.safe_dump(
            {
                "target": {"column": "target"},
                "validation": {"input": {"required_columns": ["definitely_missing_column"]}},
            }
        )
    )
    result = run_cli(
        [
            "run",
            "--config",
            str(bad_cfg),
            "--input",
            str(project_csv),
            "--output",
            str(tmp_path / "o"),
        ]
    )
    assert result.returncode == 3
    assert "definitely_missing_column" in result.stdout + result.stderr


def test_cli_exit_code_2_on_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("totally_unknown_section: {}")
    result = run_cli(["run", "--config", str(bad), "--input", "x", "--output", str(tmp_path / "o")])
    assert result.returncode == 2


def test_cli_exit_code_3_on_missing_input(tmp_path: Path) -> None:
    result = run_cli(
        ["run", "--input", str(tmp_path / "does_not_exist"), "--output", str(tmp_path / "o")]
    )
    assert result.returncode == 3


def test_cli_validate_success_and_failure(
    tmp_path: Path, project_csv: Path, good_cfg: Path
) -> None:
    ok = run_cli(["validate", "--config", str(good_cfg), "--input", str(project_csv)])
    assert ok.returncode == 0

    strict = tmp_path / "strict.yaml"
    strict.write_text(
        yaml.safe_dump(
            {
                "target": {"column": "target"},
                "validation": {"input": {"max_null_rate": {"*": 0.01}}},
            }
        )
    )
    bad = run_cli(["validate", "--config", str(strict), "--input", str(project_csv)])
    assert bad.returncode == 3


def test_cli_audit_command(tmp_path: Path, project_csv: Path, good_cfg: Path) -> None:
    out = tmp_path / "audit_out"
    result = run_cli(
        ["audit", "--config", str(good_cfg), "--input", str(project_csv), "--output", str(out)]
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert (out / "audit" / "audit.json").exists()
    assert (out / "audit" / "audit.md").exists()
    assert not (out / "data").exists()  # audit command writes no partitions


def test_cli_synthesize_command(tmp_path: Path) -> None:
    out = tmp_path / "syn"
    result = run_cli(["synthesize", "--rows", "100", "--seed", "3", "--out", str(out)])
    assert result.returncode == 0, result.stderr[-2000:]
    csv = out / "synthetic_messy.csv"
    assert csv.exists()
    frame = pl.read_csv(csv, infer_schema=False)
    assert frame.height > 100  # duplicates injected on top of the base rows


def test_cli_help_lists_commands() -> None:
    result = run_cli(["--help"])
    assert result.returncode == 0
    for command in ("run", "audit", "validate", "synthesize"):
        assert command in result.stdout


def test_cli_export_csv_flag(tmp_path: Path, project_csv: Path, good_cfg: Path) -> None:
    out = tmp_path / "csv_out"
    result = run_cli(
        [
            "run",
            "--config",
            str(good_cfg),
            "--input",
            str(project_csv),
            "--output",
            str(out),
            "--export-csv",
        ]
    )
    assert result.returncode == 0, result.stderr[-2000:]
    for name in ("train", "val", "test"):
        assert (out / "data" / f"{name}.csv").exists()


def test_cli_env_seed_override(tmp_path: Path, project_csv: Path, good_cfg: Path) -> None:
    out = tmp_path / "env_out"
    result = run_cli(
        ["run", "--config", str(good_cfg), "--input", str(project_csv), "--output", str(out)],
        env_extra={"NUMPREP_SEED": "123"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    manifest = json.loads((out / "artifacts" / "manifest.json").read_text())
    assert manifest["deterministic"]["seed"] == 123


def test_outputs_have_no_nan_inf_and_consistent_schema(
    tmp_path: Path, project_csv: Path, good_cfg: Path
) -> None:
    out = tmp_path / "out"
    result = run_cli(
        ["run", "--config", str(good_cfg), "--input", str(project_csv), "--output", str(out)]
    )
    assert result.returncode == 0, result.stderr[-2000:]
    frames = {
        name: pl.read_parquet(out / "data" / f"{name}.parquet") for name in ("train", "val", "test")
    }
    reference_columns = frames["train"].columns
    for _name, frame in frames.items():
        assert frame.columns == reference_columns
        numeric = frame.select(pl.selectors.float())
        assert numeric.select(pl.all().is_nan().any()).row(0) == tuple(
            [False] * len(numeric.columns)
        )
        assert numeric.select(pl.all().is_infinite().any()).row(0) == tuple(
            [False] * len(numeric.columns)
        )


def test_manifest_hashes_match_files(tmp_path: Path, project_csv: Path, good_cfg: Path) -> None:
    out = tmp_path / "out"
    result = run_cli(
        ["run", "--config", str(good_cfg), "--input", str(project_csv), "--output", str(out)]
    )
    assert result.returncode == 0
    import hashlib

    manifest = json.loads((out / "artifacts" / "manifest.json").read_text())
    for name, info in manifest["deterministic"]["output_files"].items():
        digest = hashlib.sha256((out / "data" / f"{name}.parquet").read_bytes()).hexdigest()
        assert digest == info["sha256"]
