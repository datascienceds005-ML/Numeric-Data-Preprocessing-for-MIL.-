"""In-process unit tests for the CLI exit-code mapping, logging, and fs helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from numprep import cli
from numprep.errors import (
    EXIT_CONFIG,
    EXIT_INPUT_CONTRACT,
    EXIT_INTERRUPTED,
    EXIT_OUTPUT_CONTRACT,
    EXIT_UNEXPECTED,
    ConfigError,
    InputContractError,
    OutputContractError,
    PipelineError,
)
from numprep.fsutils import atomic_write_json, atomic_write_text, canonical_json, sha256_file
from numprep.jsonlog import JsonFormatter, _parse_level, log_event, setup_logging, stage
from numprep.sampling import strided_sample_indices

# ---------------------------------------------------------------------------
# CLI exit-code mapping (in-process: coverage-visible; subprocess tests cover reality)
# ---------------------------------------------------------------------------


def _run_main_with(app_exception: BaseException, monkeypatch: pytest.MonkeyPatch) -> int:
    """Invoke cli.main() with a stubbed app that raises the given exception."""

    def fake_app() -> None:
        raise app_exception

    monkeypatch.setattr(cli, "app", fake_app)
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    return int(excinfo.value.code)


def test_main_maps_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(ConfigError("bad"), monkeypatch) == EXIT_CONFIG == 2


def test_main_maps_input_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(InputContractError("bad"), monkeypatch) == EXIT_INPUT_CONTRACT == 3


def test_main_maps_output_contract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(OutputContractError("bad"), monkeypatch) == EXIT_OUTPUT_CONTRACT == 4


def test_main_maps_pipeline_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(PipelineError("boom"), monkeypatch) == EXIT_UNEXPECTED == 5


def test_main_maps_generic_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(RuntimeError("boom"), monkeypatch) == EXIT_UNEXPECTED == 5


def test_main_maps_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with(KeyboardInterrupt(), monkeypatch) == EXIT_INTERRUPTED == 130


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_json_formatter_payload() -> None:
    logger = setup_logging("ERROR")
    formatter = JsonFormatter()
    record = logger.makeRecord("numprep", logging.INFO, "p", 1, "event_name", (), None)
    record.rows = 5
    payload = json.loads(formatter.format(record))
    assert payload["msg"] == "event_name"
    assert payload["rows"] == 5
    assert payload["level"] == "INFO"


def test_parse_level_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        _parse_level("LOUD")


def test_stage_context_publishes_fields_and_duration(tmp_path: Path) -> None:
    logger = setup_logging("ERROR")
    durations: dict[str, float] = {}
    with stage(logger, "demo", durations) as published:
        published["rows"] = 3
    assert "demo" in durations and durations["demo"] >= 0


def test_stage_error_reraises_and_logs(tmp_path: Path) -> None:
    logger = setup_logging("ERROR")
    with pytest.raises(RuntimeError), stage(logger, "boom"):
        raise RuntimeError("x")


def test_log_event_exc_info_path() -> None:
    logger = setup_logging("ERROR")
    try:
        raise ValueError("boom")
    except ValueError:
        log_event(logger, logging.ERROR, "caught", exc_info=True)


# ---------------------------------------------------------------------------
# fsutils + sampling
# ---------------------------------------------------------------------------


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.json"
    atomic_write_json(path, {"b": 1, "a": [1, 2]})
    content = path.read_text()
    assert json.loads(content) == {"a": [1, 2], "b": 1}
    assert content.index('"a"') < content.index('"b"')  # sorted keys
    # no temp files left behind
    assert [p.name for p in path.parent.iterdir()] == ["out.json"]


def test_atomic_write_text(tmp_path: Path) -> None:
    path = tmp_path / "t.md"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_canonical_json_stable() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_strided_sample_indices() -> None:
    assert strided_sample_indices(10, 100).tolist() == list(range(10))
    sampled = strided_sample_indices(1000, 100)
    assert len(sampled) == 100
    assert sampled[0] == 0
    assert list(sampled) == sorted(sampled)  # deterministic, ordered
    assert strided_sample_indices(0, 10).tolist() == []
