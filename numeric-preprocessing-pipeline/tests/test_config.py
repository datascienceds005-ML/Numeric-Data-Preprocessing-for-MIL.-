"""Configuration system tests: defaults, overlays, cross-field rules, error quality."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from numprep.config import CliOverrides, config_from_dict, load_config
from numprep.errors import ConfigError


def test_zero_config_defaults_are_valid() -> None:
    cfg = load_config()
    assert cfg.run.seed == 42
    assert cfg.split.strategy == "random"
    assert cfg.split.ratios == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert cfg.preprocessing.imputation.strategy == "median"
    assert cfg.output.compression == "zstd"


def test_unknown_key_rejected_with_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"spli": {"strategy": "random"}}))
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    message = str(excinfo.value)
    assert "spli" in message
    assert "unknown" in message.lower() or "extra" in message.lower()


def test_yaml_file_loading(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"seed": 7},
                "target": {"column": "y"},
                "split": {"strategy": "stratified"},
            }
        )
    )
    cfg = load_config(path)
    assert cfg.run.seed == 7
    assert cfg.target.column == "y"
    assert cfg.split.strategy == "stratified"


def test_missing_config_file_is_config_error() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("/nonexistent/config.yaml"))


def test_malformed_yaml_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("run: [unclosed")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMPREP_SEED", "99")
    monkeypatch.setenv("NUMPREP_N_JOBS", "4")
    cfg = load_config()
    assert cfg.run.seed == 99
    assert cfg.run.n_jobs == 4


def test_invalid_env_seed_is_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMPREP_SEED", "not-a-number")
    with pytest.raises(ConfigError, match="NUMPREP_SEED"):
        load_config()


def test_cli_overrides_win_over_yaml_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"run": {"seed": 1}}))
    monkeypatch.setenv("NUMPREP_SEED", "2")
    cfg = load_config(path, CliOverrides(seed=3))
    assert cfg.run.seed == 3


def test_cross_field_chronological_requires_time_column() -> None:
    with pytest.raises(ConfigError, match="time_column"):
        config_from_dict({"split": {"strategy": "chronological"}})


def test_cross_field_group_requires_group_columns() -> None:
    with pytest.raises(ConfigError, match="group_columns"):
        config_from_dict({"split": {"strategy": "group"}})


def test_cross_field_stratified_requires_target() -> None:
    with pytest.raises(ConfigError, match="target"):
        config_from_dict({"split": {"strategy": "stratified"}})


def test_cross_field_temporal_requires_chronological() -> None:
    with pytest.raises(ConfigError, match="chronological"):
        config_from_dict({"cleaning": {"temporal": {"enabled": True, "lags": [1]}}})


def test_ratio_keys_and_bounds_validated() -> None:
    with pytest.raises(ConfigError, match="exactly the keys"):
        config_from_dict({"split": {"ratios": {"train": 0.8, "val": 0.1, "holdout": 0.1}}})
    with pytest.raises(ConfigError, match=r"ratios\[.test.\]"):
        config_from_dict({"split": {"ratios": {"train": 0.7, "val": 0.15, "test": 0.0}}})


def test_thousands_separator_must_differ_from_delimiter() -> None:
    with pytest.raises(ConfigError, match="delimiter"):
        config_from_dict({"input": {"delimiter": ",", "thousands_separator": ","}})


def test_outlier_quantile_ordering_validated() -> None:
    with pytest.raises(ConfigError, match="lower_quantile"):
        config_from_dict(
            {
                "preprocessing": {
                    "outliers": {
                        "method": "winsorize",
                        "lower_quantile": 0.9,
                        "upper_quantile": 0.1,
                    }
                }
            }
        )


def test_config_fingerprint_excludes_output_directory() -> None:
    base = config_from_dict({"output": {"directory": "a"}})
    other = config_from_dict({"output": {"directory": "b"}})
    from numprep.config import config_fingerprint

    assert config_fingerprint(base) == config_fingerprint(other)


def test_delimiter_must_be_single_character() -> None:
    with pytest.raises(ConfigError, match="single character"):
        config_from_dict({"input": {"delimiter": ";;"}})
