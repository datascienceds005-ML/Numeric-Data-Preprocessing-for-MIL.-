"""Deterministic sampling helpers shared across stages.

Sampling is always *strided* (``arr[start::step]``-style) or seeded, never
ad-hoc, so identical inputs always produce identical samples. Strided sampling
needs no RNG at all and is therefore the default for audit/contract validation;
seeded RNG sampling is used only where unbiased shuffles matter (the split).
"""

from __future__ import annotations

import numpy as np


def strided_sample_indices(n_rows: int, max_samples: int) -> np.ndarray:
    """Return a deterministic strided subset of row positions ``0..n_rows-1``.

    When ``n_rows <= max_samples`` all indices are returned; otherwise indices
    are taken with a uniform stride so the sample spans the whole range.
    """
    if n_rows <= 0:
        return np.arange(0, dtype=np.int64)
    if n_rows <= max_samples:
        return np.arange(n_rows, dtype=np.int64)
    step = int(np.ceil(n_rows / max_samples))
    return np.arange(0, n_rows, step, dtype=np.int64)
