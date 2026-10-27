"""Filesystem helpers: atomic writes, content hashing, canonical JSON.

Every artifact the pipeline writes (Parquet, JSON, Markdown) is written to a
temporary file in the destination directory and then atomically renamed via
``os.replace``. CI therefore never observes partially written outputs, and reruns
are idempotent (a failed run leaves the previous complete outputs intact).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import polars as pl

_CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8) to ``path`` atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    """Write a JSON document atomically with stable key ordering and 2-space indent."""
    text = json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=True, default=str)
    atomic_write_text(path, text + "\n")


def canonical_json(payload: Any) -> str:
    """Compact, sorted, ASCII JSON used for hashing (e.g. the config hash)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def atomic_write_parquet(
    df: pl.DataFrame,
    path: Path,
    compression: Literal["zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed"] = "zstd",
) -> None:
    """Write a Polars frame to Parquet atomically with the configured compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet.tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.write_parquet(tmp_path, compression=compression, statistics=True)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
