"""Global seeding utilities.

A single integer (``run.seed``) drives every stochastic decision: stdlib ``random``,
NumPy legacy global RNG (used by some sklearn code paths), all ``numpy.random.default_rng``
instances created by the pipeline, sklearn ``random_state`` parameters, and split
shuffles.

``PYTHONHASHSEED`` is exported for *child processes* (Docker sets it for the main
process; Python cannot retroactively re-seed its own hash randomisation). The code
never relies on set/dict iteration order for any decision-bearing computation, so
hash randomisation cannot influence outputs even when the env var is unset.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed stdlib/random, NumPy globals, and export ``PYTHONHASHSEED`` for children."""
    if seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def rng(seed: int) -> np.random.Generator:
    """Return a fresh, deterministic ``numpy.random.Generator``."""
    return np.random.default_rng(seed)
