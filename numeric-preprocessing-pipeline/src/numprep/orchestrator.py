"""Orchestrator: stage sequencing, logging, artifact writing, run bookkeeping.

Execution flow (see DESIGN §2.1)::

    ingest → validate input contract → audit → clean (stateless)
           → split → fit on TRAIN → transform val/test
           → validate output contract → write outputs + artifacts
           → manifest → decision report

Every stage is instrumented (duration, row counts, peak RSS) via
:func:`numprep.jsonlog.stage`. All writes are atomic. Timestamps appear only in
the manifest's ``runtime`` block and the log file — never in data outputs — so
identical runs produce byte-identical Parquet files and identical deterministic
manifest blocks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from numprep import audit as audit_mod
from numprep import clean as clean_mod
from numprep import contracts
from numprep import ingest as ingest_mod
from numprep import pipeline as pipeline_mod
from numprep import report as report_mod
from numprep import split as split_mod
from numprep.artifacts import (
    build_manifest,
    extract_fitted_stats,
    feature_catalog_document,
    file_info,
    finalize_partition_frame,
)
from numprep.config import NumPrepConfig
from numprep.fsutils import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    sha256_file,
)
from numprep.jsonlog import LoggerType, log_event, peak_rss_mb, setup_logging, stage
from numprep.seed import set_global_seed

_INFO = 20  # logging.INFO, kept as a literal to avoid an import cycle in helpers


def _utc_now_iso() -> str:
    """Current UTC time as a stable ISO-8601 string (runtime block only)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a CSV partition (LF endings, no index)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv.tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False, lineterminator="\n")
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def run_pipeline(cfg: NumPrepConfig, log_level: str = "INFO") -> dict[str, Any]:
    """Execute the full pipeline; returns a summary dict (rows, hashes, warnings).

    The function is deliberately side-effect-ordered: nothing is written until
    the stage that owns it, and every write is atomic, so an interrupted run
    leaves the previous complete outputs untouched (idempotent reruns).
    """
    set_global_seed(cfg.run.seed)
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger: LoggerType = setup_logging(log_level, log_file=output_dir / "logs" / "run.jsonl")
    started_at = _utc_now_iso()
    durations: dict[str, float] = {}
    warnings: list[str] = []

    # S1 — ingest ---------------------------------------------------------------
    with stage(logger, "ingest", durations) as published:
        ingest_result = ingest_mod.ingest(cfg, output_dir, logger)
        published["rows"] = ingest_result.rows
        published["columns"] = len(ingest_result.columns)
        published["files"] = len(ingest_result.files)
        published["skipped"] = len(ingest_result.skipped)

    # S2 — input contract --------------------------------------------------------
    with stage(logger, "validate_input", durations) as published:
        input_contract_summary = contracts.validate_input_contract(ingest_result.frame, cfg)
        published["violations"] = input_contract_summary.get("violations", 0)

    # S3 — audit -------------------------------------------------------------------
    with stage(logger, "audit", durations) as published:
        audit_result = audit_mod.audit_frame(ingest_result.frame, cfg)
        audit_dir = output_dir / "audit"
        atomic_write_json(audit_dir / "audit.json", audit_result.to_dict())
        atomic_write_text(audit_dir / "audit.md", audit_result.to_markdown())
        published["leak_flags"] = len(audit_result.leakage_flags)
        published["corr_pairs"] = len(audit_result.high_corr_pairs)
    warnings.extend(audit_result.warnings)

    # S4 — stateless cleaning ---------------------------------------------------------
    with stage(logger, "clean", durations) as published:
        cleaned, clean_stats = clean_mod.clean_stateless(ingest_result.frame, cfg)
        published["rows_in"] = clean_stats.rows_in
        published["rows_out"] = clean_stats.rows_out
        published["duplicates_removed"] = clean_stats.duplicates_removed

    # Polars → pandas boundary (documented in DESIGN §2.2)
    frame_pd = cleaned.to_pandas()

    # S5 — split ---------------------------------------------------------------------
    with stage(logger, "split", durations) as published:
        split_result = split_mod.split_data(frame_pd, cfg)
        published.update({f"rows_{name}": size for name, size in split_result.meta.sizes.items()})
    warnings.extend(split_result.meta.warnings)

    # S6 — fit on train only -----------------------------------------------------------
    with stage(logger, "fit", durations) as published:
        pipe = pipeline_mod.build_pipeline(cfg)
        pipeline_mod.fit_preprocessor(pipe, split_result.train, cfg)
        published["columns_out"] = len(pipeline_mod.feature_names(pipe))

    # S7 — transform val/test -----------------------------------------------------------
    with stage(logger, "transform", durations):
        transformed: dict[str, pd.DataFrame] = {}
        for name, partition in split_result.partitions().items():
            transformed[name] = pipeline_mod.transform_partition(pipe, partition, cfg, name)

    # Reattach target + provenance --------------------------------------------------------
    target_column = cfg.target.column
    final_frames: dict[str, pd.DataFrame] = {}
    for name, partition in split_result.partitions().items():
        final_frames[name] = finalize_partition_frame(transformed[name], partition, target_column)

    # S8 — output contract ---------------------------------------------------------------
    with stage(logger, "validate_output", durations) as published:
        output_contract_summary = contracts.validate_output_contract(
            final_frames, target_column, cfg
        )
        published["partitions"] = output_contract_summary["partitions"]

    # S9 — write outputs ------------------------------------------------------------------
    with stage(logger, "write_outputs", durations):
        data_dir = output_dir / "data"
        output_paths: dict[str, Path] = {}
        for name in ("train", "val", "test"):
            parquet_path = data_dir / f"{name}.parquet"
            atomic_write_parquet(
                pl.from_pandas(final_frames[name]),
                parquet_path,
                compression=cfg.output.compression,
            )
            output_paths[name] = parquet_path
            if cfg.output.export_csv:
                _atomic_write_csv(final_frames[name], data_dir / f"{name}.csv")

        artifacts_dir = output_dir / "artifacts"
        preprocessor_path = artifacts_dir / "preprocessor.joblib"
        pipeline_mod.save_preprocessor(pipe, preprocessor_path)
        atomic_write_json(artifacts_dir / "fitted_stats.json", extract_fitted_stats(pipe))
        atomic_write_json(artifacts_dir / "feature_catalog.json", feature_catalog_document(pipe))

    # S10 — manifest ------------------------------------------------------------------------
    with stage(logger, "manifest", durations):
        outputs_info = {name: file_info(path) for name, path in output_paths.items()}
        manifest = build_manifest(
            cfg=cfg,
            ingest_result=ingest_result,
            split_meta=split_result.meta,
            outputs=outputs_info,
            preprocessor_sha256=sha256_file(preprocessor_path),
            peak_rss_mb=peak_rss_mb(),
            started_at_utc=started_at,
            finished_at_utc=_utc_now_iso(),
            stage_durations_s=durations,
        )
        manifest_path = artifacts_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest)

    # S11 — decision report --------------------------------------------------------------------
    with stage(logger, "report", durations):
        ctx = report_mod.RunContext(
            seed=cfg.run.seed,
            started_at_utc=started_at,
            finished_at_utc=_utc_now_iso(),
            ingest=ingest_result,
            audit=audit_result,
            clean=clean_stats,
            split_meta=split_result.meta,
            fitted_stats=extract_fitted_stats(pipe),
            output_files={name: file_info(path).sha256 for name, path in output_paths.items()},
            contract_input=input_contract_summary,
            contract_output=output_contract_summary,
            manifest_sha256=sha256_file(manifest_path),
            config_summary={"seed": cfg.run.seed},
        )
        atomic_write_text(output_dir / "reports" / "summary.md", report_mod.render_summary(ctx))

    rows_summary = {name: len(frame) for name, frame in final_frames.items()}
    log_event(
        logger,
        _INFO,
        "run_complete",
        output_dir=str(output_dir),
        rows=rows_summary,
    )
    return {
        "output_dir": str(output_dir),
        "rows": rows_summary,
        "output_hashes": {name: file_info(path).sha256 for name, path in output_paths.items()},
        "manifest": manifest,
        "durations": dict(durations),
        "warnings": warnings,
    }


def run_audit_only(cfg: NumPrepConfig, log_level: str = "INFO") -> dict[str, Any]:
    """Ingest → validate → audit; writes snapshot + audit reports, nothing else."""
    set_global_seed(cfg.run.seed)
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_level, log_file=output_dir / "logs" / "run.jsonl")

    with stage(logger, "ingest", None) as published:
        ingest_result = ingest_mod.ingest(cfg, output_dir, logger)
        published["rows"] = ingest_result.rows
    with stage(logger, "validate_input", None):
        contracts.validate_input_contract(ingest_result.frame, cfg)
    with stage(logger, "audit", None) as published:
        audit_result = audit_mod.audit_frame(ingest_result.frame, cfg)
        audit_dir = output_dir / "audit"
        atomic_write_json(audit_dir / "audit.json", audit_result.to_dict())
        atomic_write_text(audit_dir / "audit.md", audit_result.to_markdown())
        published["leak_flags"] = len(audit_result.leakage_flags)

    log_event(logger, _INFO, "audit_complete", rows=ingest_result.rows, audit_dir=str(audit_dir))
    return {"rows": ingest_result.rows, "audit_dir": str(audit_dir)}


def run_validation_only(cfg: NumPrepConfig, log_level: str = "INFO") -> dict[str, Any]:
    """Config + input-contract validation only; no artifacts are written."""
    set_global_seed(cfg.run.seed)
    logger = setup_logging(log_level)
    # Validate never persists anything: snapshot writing is disabled on a copy.
    validation_cfg = cfg.model_copy(deep=True)
    validation_cfg.input.snapshot = False

    with stage(logger, "ingest", None) as published:
        ingest_result = ingest_mod.ingest(validation_cfg, Path(cfg.output.directory), logger)
        published["rows"] = ingest_result.rows
    with stage(logger, "validate_input", None) as published:
        summary = contracts.validate_input_contract(ingest_result.frame, cfg)
        published["violations"] = summary.get("violations", 0)

    log_event(
        logger,
        _INFO,
        "validation_complete",
        rows=ingest_result.rows,
        ceilings_checked=summary.get("ceilings_checked", 0),
        ranges_checked=summary.get("ranges_checked", 0),
    )
    return summary
