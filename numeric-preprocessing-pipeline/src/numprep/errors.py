"""Exception hierarchy and process exit codes.

Every recoverable failure in the pipeline maps to one specific exception, and each
exception carries the process exit code that ``cli.main`` will use. Exit codes are
documented in README §6 and docs/DESIGN.md §10.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_CONFIG = 2
EXIT_INPUT_CONTRACT = 3
EXIT_OUTPUT_CONTRACT = 4
EXIT_UNEXPECTED = 5
EXIT_INTERRUPTED = 130


class NumPrepError(Exception):
    """Base class for all numprep errors.

    Carries the exit code the CLI should terminate with so that CI gates can
    distinguish configuration mistakes from contract violations from bugs.
    """

    exit_code: int = EXIT_UNEXPECTED

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConfigError(NumPrepError):
    """Invalid, missing, or inconsistent configuration (YAML, env, CLI overlay)."""

    exit_code = EXIT_CONFIG


class InputContractError(NumPrepError):
    """Input data violated the declared contract, or inputs could not be read."""

    exit_code = EXIT_INPUT_CONTRACT


class OutputContractError(NumPrepError):
    """Produced partitions violated the output contract (NaN/inf, schema drift)."""

    exit_code = EXIT_OUTPUT_CONTRACT


class PipelineError(NumPrepError):
    """Unexpected processing failure inside a pipeline stage."""

    exit_code = EXIT_UNEXPECTED


class LeakageError(PipelineError):
    """A train-only fitting API was called with a frame that is not the train partition.

    This is a programming/API-misuse guard: the pipeline makes leakage hard to
    commit by construction, and this error is the loud failure path.
    """

    exit_code = EXIT_UNEXPECTED
