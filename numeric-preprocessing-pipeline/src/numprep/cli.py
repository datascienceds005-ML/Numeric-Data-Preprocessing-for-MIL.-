"""Typer CLI: ``numprep run | audit | validate | synthesize``.

Exit codes (see README §6 and docs/DESIGN.md §10):

* ``0`` success · ``2`` configuration/CLI usage error · ``3`` input contract
  failure · ``4`` output contract failure · ``5`` unexpected error ·
  ``130`` SIGINT.

The CLI only parses flags and maps exceptions to exit codes; all behavior lives
in the orchestrator and stage modules (no business logic here).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from numprep.config import CliOverrides, load_config
from numprep.errors import EXIT_INTERRUPTED, EXIT_UNEXPECTED, NumPrepError
from numprep.jsonlog import log_event, setup_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Numeric data preprocessing pipeline: audit messy CSVs, clean, split, "
        "fit on train only, and export model-ready train/val/test partitions."
    ),
)


@app.command()
def run(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file; omit to use built-in defaults."
    ),
    input_paths: list[Path] | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Input CSV file, directory, or glob; repeatable; overrides config.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory; overrides config."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Global seed override."),
    n_jobs: int | None = typer.Option(
        None, "--n-jobs", help="Worker count for parallel-capable steps."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING, or ERROR."),
    export_csv: bool | None = typer.Option(
        None,
        "--export-csv/--no-export-csv",
        help="Also write CSV copies of the partitions.",
    ),
) -> None:
    """Run the full pipeline: ingest → audit → clean → split → fit → export."""
    overrides = CliOverrides(
        input_paths=tuple(str(path) for path in input_paths) if input_paths else None,
        output_dir=str(output) if output else None,
        seed=seed,
        n_jobs=n_jobs,
        export_csv=export_csv,
    )
    cfg = load_config(config, overrides)
    from numprep.orchestrator import run_pipeline

    run_pipeline(cfg, log_level)


@app.command()
def audit(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file; omit to use built-in defaults."
    ),
    input_paths: list[Path] | None = typer.Option(
        None, "--input", "-i", help="Input CSV file, directory, or glob; repeatable."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Directory for audit reports and the snapshot."
    ),
    seed: int | None = typer.Option(None, "--seed", help="Global seed override."),
    log_level: str = typer.Option("INFO", "--log-level", help="Log verbosity."),
) -> None:
    """Ingest and audit only: writes snapshot + audit.json/audit.md."""
    overrides = CliOverrides(
        input_paths=tuple(str(path) for path in input_paths) if input_paths else None,
        output_dir=str(output) if output else None,
        seed=seed,
    )
    cfg = load_config(config, overrides)
    from numprep.orchestrator import run_audit_only

    run_audit_only(cfg, log_level)


@app.command()
def validate(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file; omit to use built-in defaults."
    ),
    input_paths: list[Path] | None = typer.Option(
        None, "--input", "-i", help="Input CSV file, directory, or glob; repeatable."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Log verbosity."),
) -> None:
    """Validate config + input contract only; writes nothing; exit 3 on violation."""
    overrides = CliOverrides(
        input_paths=tuple(str(path) for path in input_paths) if input_paths else None,
    )
    cfg = load_config(config, overrides)
    from numprep.orchestrator import run_validation_only

    run_validation_only(cfg, log_level)


@app.command()
def synthesize(
    rows: int = typer.Option(5_000, "--rows", "-n", min=30, help="Row count (>= 30)."),
    seed: int = typer.Option(42, "--seed", help="Generator seed."),
    task: str = typer.Option(
        "regression", "--task", help="Target type: 'regression' or 'classification'."
    ),
    out: Path = typer.Option("data/raw", "--out", "-o", help="Output directory for the CSV."),
    log_level: str = typer.Option("INFO", "--log-level", help="Log verbosity."),
) -> None:
    """Generate a seeded messy dataset CSV for demos, tests, and smoke runs."""
    setup_logging(log_level)
    logger = logging.getLogger("numprep")
    from numprep.synthesize import generate_messy_dataset, save_messy_csv

    frame = generate_messy_dataset(rows=rows, seed=seed, task=task)
    path = save_messy_csv(frame, out / "synthetic_messy.csv")
    log_event(
        logger,
        logging.INFO,
        "synthesize_complete",
        rows=len(frame),
        columns=len(frame.columns),
        path=str(path),
        task=task,
        seed=seed,
    )


def main() -> None:
    """Console-script entry point: map exceptions to documented exit codes."""
    try:
        app()
    except NumPrepError as exc:
        logger = setup_logging("ERROR")
        log_event(
            logger,
            logging.ERROR,
            "run_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            exit_code=exc.exit_code,
        )
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        sys.exit(EXIT_INTERRUPTED)
    except Exception as exc:
        logger = setup_logging("ERROR")
        log_event(
            logger,
            logging.ERROR,
            "unexpected_failure",
            error=f"{type(exc).__name__}: {exc}",
            exc_info=True,
            exit_code=EXIT_UNEXPECTED,
        )
        sys.exit(EXIT_UNEXPECTED)


if __name__ == "__main__":  # pragma: no cover
    main()
