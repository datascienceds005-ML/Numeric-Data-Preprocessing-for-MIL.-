"""Shared fixtures and helpers for the numprep test suite."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from numprep.config import NumPrepConfig, load_config
from numprep.synthesize import generate_messy_dataset, save_messy_csv


@pytest.fixture()
def rng_gen() -> np.random.Generator:
    """Deterministic generator for test data."""
    return np.random.default_rng(7)


@pytest.fixture()
def toy_frame(rng_gen: np.random.Generator) -> pd.DataFrame:
    """Small controlled frame with nulls, a constant, an ID, and a correlated pair."""
    n = 90
    a = rng_gen.normal(10.0, 2.0, n)
    a[::9] = np.nan  # ~11% nulls
    b = rng_gen.exponential(1.0, n)
    frame = pd.DataFrame(
        {
            "id": np.arange(n, dtype="int64"),
            "a": a,
            "b": b,
            "c": 0.99 * pd.Series(a).fillna(10.0).to_numpy() + rng_gen.normal(0, 0.01, n),
            "const": np.full(n, 3.14),
            "cat": rng_gen.choice(["x", "y", "z"], n, p=[0.5, 0.3, 0.2]),
            "target": rng_gen.normal(0.0, 1.0, n),
        }
    )
    return frame


@pytest.fixture()
def toy_polars(toy_frame: pd.DataFrame) -> pl.DataFrame:
    """The toy frame as Polars (post-ingest shape, with row index)."""
    return pl.from_pandas(toy_frame).with_row_index("__row_index__")


@pytest.fixture()
def cfg() -> NumPrepConfig:
    """Default config with the toy target wired in."""
    config = load_config()
    config.target.column = "target"
    return config


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    """A 300-row seeded messy dataset on disk."""
    frame = generate_messy_dataset(rows=300, seed=11, task="regression")
    return save_messy_csv(frame, tmp_path / "raw" / "synthetic.csv")


def run_cli(
    args: list[str],
    env_extra: dict[str, str] | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a fresh process (true exit-code testing)."""
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "numprep", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


def make_cfg(sections: dict[str, Any] | None = None) -> NumPrepConfig:
    """Build a config from a raw section dict (validated exactly like production)."""
    import yaml

    text = yaml.safe_dump(sections or {})
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".yaml")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        return load_config(tmp_path)
    finally:
        os.unlink(tmp_path)
