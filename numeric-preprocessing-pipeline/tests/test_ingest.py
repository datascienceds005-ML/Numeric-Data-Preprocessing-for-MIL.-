"""Ingestion tests: path resolution, messy-CSV edge cases, coercion, downcasting."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from numprep.config import load_config
from numprep.errors import InputContractError, PipelineError
from numprep.ingest import ROW_INDEX_COL, coerce_schema, ingest, resolve_input_files
from numprep.jsonlog import setup_logging


@pytest.fixture()
def logger():
    return setup_logging("ERROR")


def test_resolve_directory_and_glob(tmp_path: Path) -> None:
    (tmp_path / "b.csv").write_text("a\n1\n")
    (tmp_path / "a.csv").write_text("a\n1\n")
    (tmp_path / "notes.txt").write_text("a\n1\n")
    (tmp_path / "skip.xlsx").write_bytes(b"not csv")
    files = resolve_input_files([str(tmp_path)])
    names = [f.name for f in files]
    # Directory inputs pick up *.csv only; notes.txt/xlsx need explicit paths.
    assert names == ["a.csv", "b.csv"]

    explicit = resolve_input_files([str(tmp_path / "notes.txt"), str(tmp_path / "skip.xlsx")])
    assert [f.name for f in explicit] == ["notes.txt"]  # xlsx rejected by suffix filter

    files = resolve_input_files([str(tmp_path / "*.csv")])
    assert [f.name for f in files] == ["a.csv", "b.csv"]


def test_resolve_no_match_raises(tmp_path: Path) -> None:
    with pytest.raises(InputContractError, match="matched no files"):
        resolve_input_files([str(tmp_path / "*.csv")])


def test_basic_ingest_and_snapshot(tmp_path: Path, logger) -> None:
    csv = tmp_path / "raw.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.rows == 2
    assert result.snapshot_path is not None and result.snapshot_path.exists()
    assert ROW_INDEX_COL in result.columns
    assert result.frame.schema["a"] == pl.Int64


def test_empty_and_header_only_files_skipped(tmp_path: Path, logger) -> None:
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "header.csv").write_text("a,b\n")
    (tmp_path / "good.csv").write_text("a,b\n1,2\n")
    cfg = load_config()
    cfg.input.paths = [str(tmp_path)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.rows == 1
    assert {s.reason for s in result.skipped} == {
        "empty file (no header, no rows)",
        "header-only file (0 data rows)",
    }


def test_all_files_empty_raises(tmp_path: Path, logger) -> None:
    (tmp_path / "empty.csv").write_text("")
    cfg = load_config()
    cfg.input.paths = [str(tmp_path)]
    with pytest.raises(InputContractError, match="no usable CSV rows"):
        ingest(cfg, tmp_path / "out", logger)


def test_inconsistent_columns_across_files_union(tmp_path: Path, logger) -> None:
    (tmp_path / "left.csv").write_text("a,b\n1,2\n")
    (tmp_path / "right.csv").write_text("a,c\n3,4\n")
    cfg = load_config()
    cfg.input.paths = [str(tmp_path)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.rows == 2
    assert {"a", "b", "c"}.issubset(set(result.columns))
    # The column absent from each file is null there.
    col_b = result.frame["b"].to_list()
    col_c = result.frame["c"].to_list()
    assert (col_b[0] is not None and col_b[1] is None) or (
        col_b[0] is None and col_b[1] is not None
    )
    assert col_c[0] is None and col_c[1] == 4


def test_bom_stripped(tmp_path: Path, logger) -> None:
    csv = tmp_path / "bom.csv"
    csv.write_bytes(b"\xef\xbb\xbfa\n1\n2\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert "a" in result.columns


def test_non_utf8_bytes_with_lossy_policy(tmp_path: Path, logger) -> None:
    csv = tmp_path / "latin.csv"
    csv.write_bytes(b"name,value\ncaf\xe9,1.5\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.input.encoding = "utf-8-lossy"
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.rows == 1


def test_latin1_encoding_decodes(tmp_path: Path, logger) -> None:
    csv = tmp_path / "latin1.csv"
    csv.write_bytes(b"name,value\ncaf\xe9,1.5\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.input.encoding = "latin-1"
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.rows == 1
    assert result.frame["value"].to_list() == [1.5]


def test_duplicate_headers_deterministically_renamed(tmp_path: Path, logger) -> None:
    csv = tmp_path / "dup.csv"
    csv.write_text("a,a,b\n1,2,3\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.duplicate_header_files == ["dup.csv"]
    assert any(c.endswith("_duplicated_0") for c in result.columns)


def test_mixed_numeric_strings_coerced(tmp_path: Path, logger) -> None:
    csv = tmp_path / "mixed.csv"
    csv.write_text('v\n"1,234.5"\n 42 \n(3.5)\n$7\n5.25%\n""\n')
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.input.delimiter = "|"
    cfg.input.thousands_separator = ","
    result = ingest(cfg, tmp_path / "out", logger)
    column = result.frame["v"]
    assert column.dtype in (pl.Float64, pl.Float32)
    values = sorted(x for x in column.to_list() if x is not None)
    assert values == pytest.approx([-3.5, 5.25, 7.0, 42.0, 1234.5])


def test_junk_heavy_column_stays_string(tmp_path: Path, logger) -> None:
    csv = tmp_path / "text.csv"
    csv.write_text("v\napple\nbanana\ncherry\n4\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["v"] == pl.String
    coercion = {c.column: c for c in result.coercions}
    assert coercion["v"].method == "kept_string"


def test_boolean_tokens_mapped(tmp_path: Path, logger) -> None:
    csv = tmp_path / "bool.csv"
    csv.write_text("flag\ntrue\nfalse\nTRUE\nNo\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["flag"] == pl.Boolean


def test_all_null_column_kept_as_float(tmp_path: Path, logger) -> None:
    csv = tmp_path / "null.csv"
    csv.write_text("v,w\nNA,1\nNA,2\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["v"] in (pl.Float64, pl.Float32)
    assert result.frame["v"].null_count() == 2


def test_dtype_override_forced(tmp_path: Path, logger) -> None:
    csv = tmp_path / "ovr.csv"
    csv.write_text("v\n1\n2\nx\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.input.dtypes = {"v": "float64"}
    cfg.run.downcast = "off"
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["v"] == pl.Float64
    assert result.frame["v"].null_count() == 1  # junk became null


def test_extreme_magnitude_blocks_downcast(tmp_path: Path, logger) -> None:
    csv = tmp_path / "big.csv"
    csv.write_text("v\n1e39\n-1e39\n1e-45\n1.0\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["v"] == pl.Float64  # guard kept float64
    notes = {n.column: n for n in result.downcasts}
    assert notes["v"].downcasted is False


def test_downcast_float32_applied(tmp_path: Path, logger) -> None:
    csv = tmp_path / "small.csv"
    csv.write_text("v\n0.5\n1.5\n-2.25\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame.schema["v"] == pl.Float32


def test_thousands_separator_config(tmp_path: Path, logger) -> None:
    csv = tmp_path / "th.csv"
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    cfg.input.delimiter = ";"
    cfg.input.thousands_separator = ","
    csv.write_text("v;w\n1,234.5;x\n")
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame["v"].to_list() == pytest.approx([1234.5])


def test_na_tokens_recognised(tmp_path: Path, logger) -> None:
    csv = tmp_path / "na.csv"
    csv.write_text("v\nNA\n?\n--\nnull\n5\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    result = ingest(cfg, tmp_path / "out", logger)
    assert result.frame["v"].null_count() == 4
    assert result.frame["v"].to_list()[-1] == 5.0


def test_coerce_schema_direct_unit() -> None:
    frame = pl.DataFrame({"s": ["1", "2", "x"], "t": ["1.5", "2.5", "3.5"]})
    cfg = load_config()
    coerced, records = coerce_schema(frame, cfg.input)
    assert coerced.schema["t"] == pl.Float64
    by_name = {r.column: r for r in records}
    assert by_name["s"].unresolved_to_null == 1
    assert by_name["t"].method == "float_cast"


def test_row_index_overflow_guard(tmp_path: Path, logger) -> None:
    csv = tmp_path / "one.csv"
    csv.write_text("v\n1\n")
    cfg = load_config()
    cfg.input.paths = [str(csv)]
    # Simulate the guard without materialising billions of rows.
    import numprep.ingest as ingest_mod

    original = ingest_mod._UINT32_MAX
    ingest_mod._UINT32_MAX = 0
    try:
        with pytest.raises(PipelineError, match="overflow"):
            ingest(cfg, tmp_path / "out", logger)
    finally:
        ingest_mod._UINT32_MAX = original
