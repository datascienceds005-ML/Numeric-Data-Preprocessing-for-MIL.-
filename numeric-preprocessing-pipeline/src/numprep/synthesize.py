"""Seeded generator of deliberately messy numeric datasets.

Pathology checklist (each injected deterministically, driven by ``seed``):

* missing values at varied rates (MCAR 3–15 %, one 97 % column, one all-null),
* ``±inf`` values and extreme outliers (1e9 magnitude),
* exact duplicate rows (~2 %),
* constant and near-constant columns,
* right-skewed and bimodal distributions,
* two correlated feature pairs (strong ≈ 0.98, moderate ≈ 0.85),
* mixed-format numeric strings (thousands separators, whitespace, parentheses
  negatives, currency symbols, percent signs),
* a low-cardinality categorical column with a rare category and a boolean with
  nulls,
* an integer ID column and an entity column (group-aware splitting),
* ordered ISO timestamps (chronological splitting / temporal features),
* a target column (regression linear-signal or imbalanced 90/10 classification)
  with ~2 % nulls.

Used by the test suite, the CI smoke run, and demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from numprep.seed import rng


def generate_messy_dataset(
    rows: int = 5_000,
    seed: int = 42,
    task: str = "regression",
) -> pd.DataFrame:
    """Generate the messy dataset. Deterministic for identical (rows, seed, task)."""
    if rows < 30:
        raise ValueError(f"rows must be >= 30, got {rows}")
    if task not in ("regression", "classification"):
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    gen = rng(seed)
    n = rows

    frame = pd.DataFrame(
        {
            "id": np.arange(n, dtype="int64"),
            "entity": np.array([f"entity_{i % 37:02d}" for i in range(n)]),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min").strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            "x_normal": gen.normal(0.0, 1.0, n),
            "x_uniform": gen.uniform(-5.0, 5.0, n),
            "x_exponential": gen.exponential(2.0, n),
            "x_bimodal": np.where(
                gen.random(n) < 0.5, gen.normal(-2.0, 0.5, n), gen.normal(2.0, 0.5, n)
            ),
            "x_lognormal": gen.lognormal(0.0, 1.0, n),
            "denominator": np.abs(gen.normal(10.0, 3.0, n)),
            "cat_low": gen.choice(
                ["alpha", "beta", "gamma", "delta", "omega"],
                n,
                p=[0.4, 0.3, 0.2, 0.09, 0.01],
            ),
            "cat_bool": gen.choice(
                np.array([True, False, None], dtype=object), n, p=[0.45, 0.45, 0.10]
            ),
            "const_col": np.full(n, 42.0),
            "near_const": np.where(gen.random(n) < 0.005, 7.0, 1.0),
            "all_null_col": np.full(n, np.nan),
            "mostly_null": np.where(gen.random(n) < 0.97, np.nan, gen.normal(0, 1, n)),
        }
    )

    frame["x_corr_strong"] = 0.98 * frame["x_normal"] + gen.normal(0.0, 0.05, n)
    frame["x_corr_moderate"] = 0.85 * frame["x_uniform"] + gen.normal(0.0, 1.0, n)
    frame["numerator"] = 3.0 * frame["denominator"] + gen.normal(0.0, 1.0, n)

    # Mixed-format numeric strings ------------------------------------------
    base = gen.normal(100.0, 25.0, n)
    styles = gen.integers(0, 5, n)
    mixed = np.empty(n, dtype=object)
    for index in range(n):
        value = float(base[index])
        style = int(styles[index])
        if style == 0:
            mixed[index] = f"{value:,.2f}"  # thousands separator
        elif style == 1:
            mixed[index] = f" {value:.3f} "  # whitespace padding
        elif style == 2 and value < 0:
            mixed[index] = f"({abs(value):.2f})"  # parenthesised negative
        elif style == 3:
            mixed[index] = f"${value:.2f}"  # currency symbol
        else:
            mixed[index] = f"{value:.4f}"
    frame["x_mixed_format"] = mixed

    percent_base = np.abs(gen.normal(15.0, 5.0, n))
    frame["x_percent_string"] = np.array([f"{value:.1f}%" for value in percent_base], dtype=object)

    # Outliers & infinities --------------------------------------------------
    outlier_idx = gen.choice(n, size=max(n // 100, 3), replace=False)
    frame.loc[outlier_idx, "x_normal"] = gen.choice([1e9, -1e9, 5e8], size=outlier_idx.size)
    inf_idx = gen.choice(n, size=max(n // 200, 2), replace=False)
    frame.loc[inf_idx, "x_exponential"] = gen.choice([np.inf, -np.inf], size=inf_idx.size)

    # Missingness -------------------------------------------------------------
    for column, rate in (
        ("x_normal", 0.03),
        ("x_uniform", 0.05),
        ("x_exponential", 0.08),
        ("x_bimodal", 0.10),
        ("x_lognormal", 0.15),
        ("x_corr_strong", 0.04),
        ("x_corr_moderate", 0.06),
        ("denominator", 0.05),
        ("numerator", 0.07),
    ):
        mask = gen.random(n) < rate
        frame.loc[mask, column] = np.nan
    frame.loc[gen.random(n) < 0.05, "cat_low"] = None

    # Target ------------------------------------------------------------------
    signal = (
        0.8 * frame["x_normal"].fillna(0.0)
        + 0.5 * frame["x_lognormal"].fillna(1.0)
        - 0.3 * frame["x_exponential"].fillna(0.0)
        + 0.25 * frame["numerator"].fillna(0.0) / frame["denominator"].fillna(1.0)
    )
    if task == "regression":
        frame["target"] = signal + gen.normal(0.0, 0.25, n)
    else:
        probability = 1.0 / (1.0 + np.exp(-0.6 * (signal - signal.mean())))
        frame["target"] = (gen.random(n) < probability).astype("int64")
        # Force class imbalance toward ~90/10 for stratification demos.
        flip = (frame["target"] == 1) & (gen.random(n) < 0.45)
        frame.loc[flip, "target"] = 0
    frame.loc[gen.random(n) < 0.02, "target"] = np.nan

    # Exact duplicates (~2 %) --------------------------------------------------
    duplicate_count = max(int(n * 0.02), 2)
    duplicate_idx = gen.choice(n, size=duplicate_count, replace=False)
    frame = pd.concat([frame, frame.iloc[duplicate_idx]], ignore_index=True)

    return frame


def save_messy_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write the dataset to CSV deterministically (LF endings, ``NA`` nulls)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", na_rep="NA")
    return path
