"""numprep — production-grade preprocessing pipeline for messy numeric datasets.

The package is organised as a strict dependency DAG:

    cli → orchestrator → (ingest, contracts, audit, clean, temporal, split,
                          pipeline, artifacts, report) → transformers → utils
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
