"""Run artifacts: manifest.json, human-readable fitted statistics, feature catalog.

The manifest is split into two blocks on purpose (DESIGN §7.1):

* ``deterministic`` — config hash, input hashes, library versions, seed, split
  sizes, output hashes. Identical input + config + image ⇒ byte-identical block.
* ``runtime`` — wall-clock timestamps, stage durations, peak RSS. Kept separate
  so data outputs and the deterministic block stay reproducible; the CI
  determinism gate compares only the deterministic block plus file hashes.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import pydantic
import sklearn

from numprep import __version__
from numprep.config import NumPrepConfig, config_fingerprint
from numprep.errors import PipelineError
from numprep.fsutils import sha256_file
from numprep.ingest import ROW_INDEX_COL, IngestResult
from numprep.split import SplitMeta
from numprep.transformers.base import FrameTransformer


@dataclass(frozen=True)
class OutputFileInfo:
    """Hash/row info for one written data file."""

    path: Path
    rows: int
    columns: int
    sha256: str


def git_sha() -> str | None:
    """Return the current git commit SHA, or ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def library_versions() -> dict[str, str]:
    """Pin the full library context of the run (deterministic per image)."""
    return {
        "python": sys.version.split()[0],
        "numprep": __version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "polars": pl.__version__,
        "pyarrow": pyarrow.__version__,
        "scikit-learn": sklearn.__version__,
        "pydantic": pydantic.__version__,
    }


def build_manifest(
    cfg: NumPrepConfig,
    ingest_result: IngestResult,
    split_meta: SplitMeta,
    outputs: dict[str, OutputFileInfo],
    preprocessor_sha256: str | None,
    peak_rss_mb: float,
    started_at_utc: str,
    finished_at_utc: str,
    stage_durations_s: dict[str, float],
) -> dict[str, Any]:
    """Assemble the two-block manifest (deterministic + runtime)."""
    deterministic: dict[str, Any] = {
        "numprep_version": __version__,
        "seed": cfg.run.seed,
        "config_sha256": _config_hash(cfg),
        "git_sha": git_sha(),
        "library_versions": library_versions(),
        "input_files": [
            {"name": f.path.name, "sha256": f.sha256, "rows": f.rows} for f in ingest_result.files
        ],
        "skipped_files": [{"name": f.path.name, "reason": f.reason} for f in ingest_result.skipped],
        "ingestion": {
            "rows": ingest_result.rows,
            "columns": len(ingest_result.columns),
            "snapshot": ingest_result.snapshot_path.name if ingest_result.snapshot_path else None,
        },
        "split": split_meta.to_dict(),
        "output_files": {
            name: {
                "name": info.path.name,
                "rows": info.rows,
                "columns": info.columns,
                "sha256": info.sha256,
            }
            for name, info in outputs.items()
        },
        "preprocessor_sha256": preprocessor_sha256,
    }
    runtime: dict[str, Any] = {
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "stage_durations_s": {k: round(v, 3) for k, v in stage_durations_s.items()},
        "peak_rss_mb": round(peak_rss_mb, 1),
    }
    return {"deterministic": deterministic, "runtime": runtime}


def _config_hash(cfg: NumPrepConfig) -> str:
    """SHA-256 of the canonical JSON fingerprint of the validated config."""
    import hashlib

    return hashlib.sha256(config_fingerprint(cfg).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Human-readable fitted statistics
# ---------------------------------------------------------------------------


def _round_floats(value: Any, places: int = 6) -> Any:
    """Round floats for readability (deterministic; statistics only, not data)."""
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return round(value, places)
    if isinstance(value, dict):
        return {key: _round_floats(item, places) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, places) for item in value]
    return value


def _step(pipe: Any, name: str) -> Any:
    """Fetch a fitted pipeline step or ``None``."""
    if name in pipe.named_steps:
        return pipe.named_steps[name]
    return None


def extract_fitted_stats(pipe: Any) -> dict[str, Any]:
    """Extract JSON-serializable, human-readable fitted statistics per step."""
    stats: dict[str, Any] = {}

    pruner: FrameTransformer | None = _step(pipe, "prune")
    if pruner is not None and hasattr(pruner, "drop_reasons_"):
        stats["column_pruning"] = {
            "dropped": dict(pruner.drop_reasons_),
            "kept_columns": list(pruner.columns_to_keep_),
        }

    indicators = _step(pipe, "indicators")
    if indicators is not None and hasattr(indicators, "indicator_columns_"):
        stats["missing_indicators"] = list(indicators.indicator_columns_)

    imputer = _step(pipe, "impute")
    if imputer is not None and hasattr(imputer, "statistics_"):
        stats["imputation"] = {
            "strategy": imputer.strategy,
            "fill_values": _round_floats(dict(imputer.statistics_)),
        }

    clipper = _step(pipe, "clip")
    if clipper is not None and hasattr(clipper, "bounds_"):
        stats["outlier_bounds"] = {
            "method": clipper.method,
            "bounds": {
                name: {"lower": _round_floats(low), "upper": _round_floats(high)}
                for name, (low, high) in (
                    (k, v) for k, v in clipper.bounds_.items() if v is not None
                )
            },
        }

    engineer = _step(pipe, "features")
    if engineer is not None and hasattr(engineer, "catalog_"):
        stats["feature_engineering"] = {
            "catalog": [entry.to_dict() for entry in engineer.catalog_],
            "engineered_fill_values": _round_floats(dict(engineer.fill_values_)),
            "dropped": dict(engineer.dropped_features_),
        }

    encoder = _step(pipe, "encode")
    if encoder is not None and hasattr(encoder, "categories_"):
        stats["encoding"] = {
            "categories": {k: list(v) for k, v in encoder.categories_.items()},
            "dropped": dict(encoder.dropped_columns_),
        }

    selector = _step(pipe, "select")
    if selector is not None and hasattr(selector, "drop_reasons_"):
        stats["selection"] = {"dropped": dict(selector.drop_reasons_)}

    scaler = _step(pipe, "scale")
    if scaler is not None and hasattr(scaler, "kinds_"):
        per_column: dict[str, Any] = {}
        for name, kind in scaler.kinds_.items():
            entry: dict[str, Any] = {"kind": kind, "reason": scaler.reasons_[name]}
            sub = scaler.scalers_.get(name)
            if sub is not None:
                entry["params"] = _round_floats(_scaler_params(sub))
            per_column[name] = entry
        stats["scaling"] = per_column

    result: dict[str, Any] = _round_floats(stats)
    return result


def _scaler_params(scaler: Any) -> dict[str, Any]:
    """Pull the descriptive parameters from any supported sub-scaler."""
    if hasattr(scaler, "lambdas_"):  # PowerTransformer
        return {"lambdas": list(np.atleast_1d(scaler.lambdas_))}
    if hasattr(scaler, "n_quantiles_"):  # QuantileTransformer
        return {"n_quantiles": int(scaler.n_quantiles_)}
    if hasattr(scaler, "center_") and hasattr(scaler, "scale_"):  # RobustScaler
        return {"center": scaler.center_, "scale": scaler.scale_}
    if hasattr(scaler, "data_min_"):  # MinMaxScaler
        return {"min": scaler.data_min_, "max": scaler.data_max_}
    if hasattr(scaler, "mean_") and hasattr(scaler, "var_"):  # StandardScaler
        return {"mean": scaler.mean_, "var": scaler.var_}
    return {}


def feature_catalog_document(pipe: Any) -> dict[str, Any]:
    """Build the standalone feature-catalog artifact (engineered features)."""
    engineer = _step(pipe, "features")
    if engineer is None or not hasattr(engineer, "catalog_"):
        return {"features": []}
    return {
        "features": [entry.to_dict() for entry in engineer.catalog_],
        "dropped": dict(engineer.dropped_features_),
        "count": len(engineer.catalog_),
    }


# ---------------------------------------------------------------------------
# Output writing helpers
# ---------------------------------------------------------------------------


def finalize_partition_frame(
    features: pd.DataFrame,
    original: pd.DataFrame,
    target_column: str | None,
) -> pd.DataFrame:
    """Reattach target + provenance column; enforce the documented column order.

    Order: engineered/retained feature columns, then the target (unchanged), then
    ``__row_index__``. Guarantees identical column order across partitions.
    """
    out = features.copy(deep=False)
    if target_column is not None:
        if target_column not in original.columns:
            raise PipelineError(f"target column {target_column!r} missing when reattaching")
        out[target_column] = original[target_column].to_numpy()
    if ROW_INDEX_COL in original.columns:
        out[ROW_INDEX_COL] = original[ROW_INDEX_COL].to_numpy()
    return out


def file_info(path: Path) -> OutputFileInfo:
    """Hash a written file and record its shape (for Parquet outputs)."""
    sha = sha256_file(path)
    rows = columns = -1
    if path.suffix == ".parquet":
        import polars as pl

        frame = pl.scan_parquet(path)
        rows = int(frame.select(pl.len()).collect().item())
        columns = len(frame.collect_schema())
    return OutputFileInfo(path=path, rows=rows, columns=columns, sha256=sha)
