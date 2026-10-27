"""Structured JSON logging with per-stage timing and peak-RSS instrumentation.

All pipeline logging goes through this module; there are no ``print`` statements
anywhere in the package. Every record is a single-line JSON object on stdout and,
when a run directory is active, appended to ``<output>/logs/run.jsonl``.

Peak RSS uses ``getrusage(RUSAGE_SELF).ru_maxrss`` — a process-level high-water
mark. It is monotonic, so per-stage *deltas* can under-report (a stage that
allocates less than a previous peak shows ~0 delta). This is documented honestly
in the benchmark report and README rather than faked with per-stage precision.
"""

from __future__ import annotations

import json
import logging
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_RESERVED_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stacklevel",
        "thread",
        "threadName",
        "taskName",
    }
)

LOGGER_NAME = "numprep"

LoggerType = logging.Logger
"""Alias for the structured logger used across the package."""


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON with merged ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def peak_rss_mb() -> float:
    """Return the process peak resident set size in MiB (platform-correct)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    # Linux reports KiB.
    return raw / 1024.0


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """(Re)initialise the ``numprep`` logger with a stdout handler and optional file sink."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(_parse_level(level))

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    return logger


def _parse_level(level: str) -> int:
    """Translate a level name into a numeric logging level with a helpful error."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"unknown log level: {level!r} (use DEBUG, INFO, WARNING or ERROR)")
    return numeric


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured event; ``fields`` are merged into the JSON payload.

    ``exc_info`` is special-cased and forwarded to the logging call itself so
    tracebacks render correctly instead of colliding with the reserved
    ``LogRecord`` attribute.
    """
    exc_info = fields.pop("exc_info", False)
    logger.log(level, event, exc_info=exc_info, extra={"event": event, **fields})


@contextmanager
def stage(
    logger: logging.Logger,
    name: str,
    durations: dict[str, float] | None = None,
    **static_fields: Any,
) -> Iterator[dict[str, Any]]:
    """Instrument a pipeline stage: logs start/done/error with duration and RSS.

    The yielded dict may be populated by the stage body (e.g. ``rows``); its keys
    are merged into the ``<name>.done`` event so per-stage row counts land in the
    structured log automatically. When ``durations`` is a dict, the stage's
    wall-clock duration is recorded under ``name`` for the manifest.
    """
    published: dict[str, Any] = {}
    rss_before = peak_rss_mb()
    started = time.perf_counter()
    log_event(logger, logging.INFO, f"{name}.start", stage=name, **static_fields)
    try:
        yield published
    except Exception:
        duration = round(time.perf_counter() - started, 3)
        log_event(
            logger,
            logging.ERROR,
            f"{name}.error",
            stage=name,
            duration_s=duration,
            peak_rss_mb=round(peak_rss_mb(), 1),
            **static_fields,
        )
        raise
    else:
        duration = round(time.perf_counter() - started, 3)
        rss_after = peak_rss_mb()
        log_event(
            logger,
            logging.INFO,
            f"{name}.done",
            stage=name,
            duration_s=duration,
            peak_rss_mb=round(rss_after, 1),
            rss_delta_mb=round(max(rss_after - rss_before, 0.0), 1),
            **published,
            **static_fields,
        )
        if durations is not None:
            durations[name] = duration
