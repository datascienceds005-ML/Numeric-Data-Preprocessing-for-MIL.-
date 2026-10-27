"""Ingestion: raw CSVs → typed Parquet snapshot via Polars streaming.

Design notes
------------
* Files/directories/globs are resolved in sorted order for determinism.
* Every file is scanned as **all-String** first (``infer_schema=False``): mixed-type
  columns, thousands separators, currency symbols, parenthesised negatives and BOMs
  are normalised by explicit string operations before a ``strict=False`` numeric
  cast. Columns with fewer castable values than ``numeric_min_castable_ratio`` stay
  String (and are later handled by the fitted encoder). Every coercion is logged.
* ``inf`` / ``-inf`` / ``Infinity`` strings cast to real infinities, which the
  stateless cleaning stage converts to nulls (so they are imputed, not propagated).
* Duplicate header names are handled by Polars deterministically (``a_duplicated_N``);
  the raw header is sniffed so the event is logged.
* Non-UTF-8 input: ``utf-8-lossy`` (replace semantics, default) via Polars, or a
  manual ``latin-1`` decode (whole-file, memory-bound; documented).
* A ``__row_index__`` (UInt32) column is appended in file+row order: it is the
  stable tie-break for chronological sorting and temporal features.
* The typed snapshot is written atomically as Parquet (zstd) — this is the artifact
  every later stage reads, so raw CSVs are touched exactly once per run.
"""

from __future__ import annotations

import csv as _csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from numprep.config import InputConfig, NumPrepConfig
from numprep.errors import InputContractError, PipelineError
from numprep.fsutils import atomic_write_parquet, sha256_file
from numprep.jsonlog import LoggerType

ROW_INDEX_COL = "__row_index__"
_WHITESPACE = " \t\r\n\u00a0"
_BOOL_TOKENS: dict[str, bool] = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
    "t": True,
    "f": False,
}
_UINT32_MAX = 2**32 - 1
_FLOAT64_DTYPE = pl.Float64
_FLOAT32_DTYPE = pl.Float32

_PAREN_RE = re.compile(r"^\((.*)\)$")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedFile:
    """A raw input file that contributed rows to the snapshot."""

    path: Path
    rows: int
    sha256: str


@dataclass(frozen=True)
class SkippedFile:
    """A raw input file that was skipped (empty or unreadable-as-CSV)."""

    path: Path
    reason: str


@dataclass(frozen=True)
class ColumnCoercion:
    """Record of how one column's dtype was determined."""

    column: str
    method: str  # boolean_map | integer_cast | float_cast | kept_string | all_null | override
    target_dtype: str
    unresolved_to_null: int


@dataclass(frozen=True)
class DowncastNote:
    """Record of a float64→float32 downcast decision for one column."""

    column: str
    downcasted: bool
    reason: str


@dataclass(frozen=True)
class IngestResult:
    """Everything downstream stages need to know about the ingestion."""

    frame: pl.DataFrame
    snapshot_path: Path | None
    files: list[ResolvedFile]
    skipped: list[SkippedFile]
    coercions: list[ColumnCoercion]
    downcasts: list[DowncastNote]
    rows: int
    columns: list[str]
    duplicate_header_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_input_files(paths: list[str]) -> list[Path]:
    """Resolve a mix of files, directories, and glob patterns into a sorted file list.

    Directory inputs pick up ``*.csv`` files; glob patterns expand relative to the
    current working directory. Duplicates are removed preserving first-seen order.
    Raises :class:`InputContractError` when nothing matches.
    """
    collected: list[Path] = []
    for raw in paths:
        candidate = Path(raw).expanduser()
        if candidate.is_dir():
            collected.extend(sorted(candidate.glob("*.csv")))
        elif any(ch in raw for ch in "*?["):
            # glob.glob supports absolute patterns (pathlib.Path.glob does not).
            import glob as _glob

            matches = sorted(Path(p) for p in _glob.glob(raw, recursive=True) if Path(p).is_file())
            if not matches:
                raise InputContractError(
                    f"input path pattern {raw!r} matched no files (cwd: {Path.cwd()})"
                )
            collected.extend(matches)
        elif candidate.is_file():
            collected.append(candidate)
        else:
            raise InputContractError(
                f"input path {raw!r} does not exist (looked for file, directory, or glob)"
            )

    seen: set[str] = set()
    unique: list[Path] = []
    for path in collected:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)

    csv_like = [p for p in unique if p.suffix.lower() in {".csv", ".tsv", ".txt"}]
    if not csv_like:
        raise InputContractError(
            f"no CSV-like inputs found among {[str(p) for p in unique] or paths!r} "
            "(accepted suffixes: .csv, .tsv, .txt)"
        )
    return csv_like


# ---------------------------------------------------------------------------
# Low-level reading
# ---------------------------------------------------------------------------


def _decode_head(path: Path, encoding: str, nbytes: int = 1 << 20) -> str:
    """Decode the first ``nbytes`` of a file with the configured encoding policy."""
    with path.open("rb") as handle:
        raw = handle.read(nbytes)
    try:
        if encoding == "latin-1":
            return raw.decode("latin-1")
        if encoding == "utf-8":
            return raw.decode("utf-8")
        return raw.decode("utf-8", errors="replace")
    except UnicodeDecodeError as exc:
        raise InputContractError(
            f"file {path} contains bytes that are not valid UTF-8 "
            f"(set input.encoding='utf-8-lossy' or 'latin-1' to tolerate them): {exc}"
        ) from exc


def _sniff_header_names(path: Path, encoding: str, delimiter: str) -> list[str] | None:
    """Return stripped header names, or ``None`` when the file is empty/header-less."""
    head = _decode_head(path, encoding)
    lines = head.splitlines()
    if not lines or not lines[0].strip():
        return None
    row = next(_csv.reader([lines[0]], delimiter=delimiter))
    return [name.strip() for name in row]


def _polars_encoding(encoding: str) -> str:
    """Map our encoding names onto Polars' two native CSV encodings."""
    if encoding == "utf-8":
        return "utf8"
    return "utf8-lossy"


def _scan_file(path: Path, input_cfg: InputConfig) -> pl.LazyFrame:
    """Lazily scan one CSV as an all-String frame with configured NA tokens."""
    kwargs: dict[str, object] = {
        "separator": input_cfg.delimiter,
        "has_header": input_cfg.has_header,
        "null_values": list(input_cfg.na_tokens),
        "infer_schema": False,
        "encoding": _polars_encoding(input_cfg.encoding),
    }
    try:
        return pl.scan_csv(path, **kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        raise InputContractError(f"failed to read CSV {path}: {exc}") from exc


def _read_file_eager(path: Path, input_cfg: InputConfig) -> pl.DataFrame:
    """Eagerly read one CSV (latin-1 path only) as an all-String frame."""
    try:
        text = path.read_bytes().decode("latin-1")
    except OSError as exc:
        raise InputContractError(f"failed to read file {path}: {exc}") from exc
    try:
        return pl.read_csv(
            io.StringIO(text),
            separator=input_cfg.delimiter,
            has_header=input_cfg.has_header,
            null_values=list(input_cfg.na_tokens),
            infer_schema=False,
        )
    except Exception as exc:
        raise InputContractError(f"failed to parse CSV {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _clean_string_series(s: pl.Series, input_cfg: InputConfig) -> pl.Series:
    """Normalise a String series: whitespace, currency, thousands, parentheses."""
    cleaned = s.str.strip_chars(_WHITESPACE)
    for symbol in input_cfg.currency_symbols:
        if symbol:
            cleaned = cleaned.str.replace_all(symbol, "", literal=True)
    if input_cfg.thousands_separator:
        cleaned = cleaned.str.replace_all(input_cfg.thousands_separator, "", literal=True)
    if input_cfg.parentheses_negative:
        cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-$1")
    return cleaned


def _try_boolean(s: pl.Series) -> pl.Series | None:
    """Map boolean-ish tokens to a Boolean series, or return ``None``."""
    lowered = s.str.to_lowercase()
    non_null = lowered.drop_nulls()
    if non_null.len() == 0:
        return None
    allowed = lowered.is_in(list(_BOOL_TOKENS))
    if not bool(allowed.fill_null(False).all()):
        return None
    return lowered.replace_strict(_BOOL_TOKENS, default=None, return_dtype=pl.Boolean)


def _coerce_single_column(
    name: str, s: pl.Series, input_cfg: InputConfig
) -> tuple[pl.Series, ColumnCoercion]:
    """Determine the best dtype for one all-String column and cast it."""
    null_count_before = s.null_count()
    non_null_before = s.len() - null_count_before

    override = input_cfg.dtypes.get(name)
    cleaned = _clean_string_series(s, input_cfg)

    if override is not None:
        target, method = _cast_with_override(cleaned, override)
        unresolved = int(target.null_count()) - null_count_before
        return target, ColumnCoercion(name, method, str(target.dtype), max(unresolved, 0))

    if non_null_before == 0:
        empty = pl.Series(name, [None] * s.len(), dtype=_FLOAT64_DTYPE)
        return empty, ColumnCoercion(name, "all_null", "Float64", 0)

    boolean = _try_boolean(cleaned)
    if boolean is not None:
        return (
            boolean.rename(name),
            ColumnCoercion(name, "boolean_map", "Boolean", 0),
        )

    as_int = cleaned.cast(pl.Int64, strict=False)
    if int(as_int.null_count()) == null_count_before:
        return as_int.rename(name), ColumnCoercion(name, "integer_cast", "Int64", 0)

    as_float = cleaned.cast(_FLOAT64_DTYPE, strict=False)
    castable = non_null_before - (int(as_float.null_count()) - null_count_before)
    ratio = castable / non_null_before
    if ratio >= input_cfg.numeric_min_castable_ratio:
        unresolved = int(as_float.null_count()) - null_count_before
        return (
            as_float.rename(name),
            ColumnCoercion(name, "float_cast", "Float64", max(unresolved, 0)),
        )

    return cleaned.rename(name), ColumnCoercion(name, "kept_string", "String", 0)


def _cast_with_override(s: pl.Series, override: str) -> tuple[pl.Series, str]:
    """Cast a cleaned String series to the explicitly configured dtype."""
    if override == "string":
        return s, "override"
    if override == "bool":
        lowered = s.str.to_lowercase()
        mapped = lowered.replace_strict(_BOOL_TOKENS, default=None, return_dtype=pl.Boolean)
        return mapped, "override"
    dtype = {
        "float32": _FLOAT32_DTYPE,
        "float64": _FLOAT64_DTYPE,
        "int64": pl.Int64,
        "int32": pl.Int32,
    }[override]
    return s.cast(dtype, strict=False), "override"


def coerce_schema(
    df: pl.DataFrame, input_cfg: InputConfig
) -> tuple[pl.DataFrame, list[ColumnCoercion]]:
    """Apply per-column dtype coercion (booleans → ints → floats → string fallback)."""
    coerced: list[ColumnCoercion] = []
    columns: list[pl.Series] = []
    for name in df.columns:
        if name == ROW_INDEX_COL:
            columns.append(df[name])
            continue
        new_series, record = _coerce_single_column(name, df[name], input_cfg)
        columns.append(new_series)
        coerced.append(record)
    return pl.DataFrame(columns), coerced


def _sample_indices(n_rows: int, max_samples: int = 200_000) -> np.ndarray:
    """Deterministic strided sample of row positions (no RNG needed)."""
    if n_rows <= max_samples:
        return np.arange(n_rows)
    step = int(np.ceil(n_rows / max_samples))
    return np.arange(0, n_rows, step)


def downcast_frame(df: pl.DataFrame, mode: str) -> tuple[pl.DataFrame, list[DowncastNote]]:
    """Float64 → Float32 with a deterministic precision guard.

    ``auto`` casts only when a strided sample survives an ``allclose`` check
    (``rtol=1e-6``); ``force`` casts unconditionally; ``off`` keeps float64.
    """
    if mode == "off" or df.height == 0:
        return df, []
    notes: list[DowncastNote] = []
    indices = _sample_indices(df.height)
    new_columns: list[pl.Series] = []
    changed = False
    for name in df.columns:
        series = df[name]
        if series.dtype != _FLOAT64_DTYPE:
            new_columns.append(series)
            continue
        sample64 = series.gather(pl.Series(indices)).to_numpy(allow_copy=True)
        with np.errstate(over="ignore"):
            sample32 = sample64.astype(np.float32)
        ok = bool(
            np.allclose(
                sample64,
                sample32.astype(np.float64),
                rtol=1e-6,
                atol=1e-9,
                equal_nan=True,
            )
        )
        if mode == "force" or ok:
            new_columns.append(series.cast(_FLOAT32_DTYPE).rename(name))
            changed = True
            notes.append(
                DowncastNote(
                    name,
                    True,
                    "forced" if mode == "force" else "precision guard passed",
                )
            )
        else:
            notes.append(DowncastNote(name, False, "precision guard failed; kept Float64"))
    if not changed:
        return df, notes
    return pl.DataFrame(new_columns), notes


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def ingest(cfg: NumPrepConfig, output_dir: Path, logger: LoggerType) -> IngestResult:
    """Run the full ingestion stage and write the typed Parquet snapshot.

    Raises :class:`InputContractError` when no usable input is found; all other
    failures surface as :class:`PipelineError`.
    """
    files = resolve_input_files(cfg.input.paths)

    lazy_frames: list[pl.LazyFrame] = []
    resolved: list[ResolvedFile] = []
    skipped: list[SkippedFile] = []
    duplicate_header_files: list[str] = []

    for path in files:
        names = _sniff_header_names(path, cfg.input.encoding, cfg.input.delimiter)
        if names is None:
            skipped.append(SkippedFile(path, "empty file (no header, no rows)"))
            continue
        if len(set(names)) != len(names) and cfg.input.has_header:
            duplicate_header_files.append(path.name)
        if cfg.input.encoding == "latin-1":
            frame = _read_file_eager(path, cfg.input)
            if frame.height == 0:
                skipped.append(SkippedFile(path, "header-only file (0 data rows)"))
                continue
            lazy_frames.append(frame.lazy())
            resolved.append(ResolvedFile(path, frame.height, sha256_file(path)))
        else:
            lazy_frame = _scan_file(path, cfg.input)
            row_count: int | None = None
            try:
                row_count = int(lazy_frame.select(pl.len()).collect().item())
            except InputContractError:
                raise
            except Exception as exc:
                raise InputContractError(f"failed to read CSV {path}: {exc}") from exc
            if row_count == 0:
                skipped.append(SkippedFile(path, "header-only file (0 data rows)"))
                continue
            lazy_frames.append(lazy_frame)
            resolved.append(ResolvedFile(path, row_count, sha256_file(path)))

    if not lazy_frames:
        raise InputContractError(
            "no usable CSV rows found: "
            + "; ".join(f"{p.name} ({r})" for p, r in [(s.path, s.reason) for s in skipped])
            or "no input files contained data"
        )

    try:
        combined = pl.concat(lazy_frames, how="diagonal").collect(engine="streaming")
    except Exception as exc:
        raise PipelineError(f"failed to concatenate input files: {exc}") from exc

    if combined.height == 0:
        raise InputContractError("combined input contains 0 rows after ingestion")
    if combined.height > _UINT32_MAX:
        raise PipelineError(
            f"input exceeds {_UINT32_MAX} rows; __row_index__ (UInt32) would overflow"
        )

    if duplicate_header_files and logger is not None:
        logger.warning(
            "duplicate_header_columns_renamed",
            extra={
                "event": "duplicate_header_columns_renamed",
                "files": duplicate_header_files,
                "note": "polars renamed duplicates deterministically as <name>_duplicated_<n>",
            },
        )

    combined = combined.with_row_index(ROW_INDEX_COL)

    combined, coercions = coerce_schema(combined, cfg.input)
    combined, downcasts = downcast_frame(combined, cfg.run.downcast)

    snapshot_path: Path | None = None
    if cfg.input.snapshot:
        snapshot_path = output_dir / "snapshot" / "data.parquet"
        atomic_write_parquet(combined, snapshot_path, compression=cfg.output.compression)

    return IngestResult(
        frame=combined,
        snapshot_path=snapshot_path,
        files=resolved,
        skipped=skipped,
        coercions=coercions,
        downcasts=downcasts,
        rows=combined.height,
        columns=list(combined.columns),
        duplicate_header_files=duplicate_header_files,
    )
