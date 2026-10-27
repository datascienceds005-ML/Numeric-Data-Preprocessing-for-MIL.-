"""Partition tagging and the train-only fitting guard.

This module is the leakage-prevention keystone (DESIGN §4.1):

* ``split.py`` is the *only* code that tags frames via :func:`tag_partition`;
* every fitted transformer and ``pipeline.fit_preprocessor`` calls
  :func:`require_train`, which raises :class:`LeakageError` unless the frame
  carries the ``TRAIN`` tag — so a raw/unsplit frame or a val/test partition
  cannot reach any ``fit`` without an immediate, loud failure;
* ``require_partition`` gives the transform path the same rigor.
"""

from __future__ import annotations

import enum

import pandas as pd

from numprep.errors import LeakageError

PARTITION_KEY = "numprep.partition"


class Partition(enum.Enum):
    """The three dataset partitions produced by the split stage."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def tag_partition(df: pd.DataFrame, role: Partition) -> pd.DataFrame:
    """Return a shallow copy of ``df`` tagged with its partition role.

    Only :mod:`numprep.split` should call this — downstream code trusts the tag.
    """
    tagged = df.copy(deep=False)
    tagged.attrs = {**tagged.attrs, PARTITION_KEY: role.value}
    return tagged


def partition_of(df: pd.DataFrame) -> Partition | None:
    """Return the partition role of a frame, or ``None`` when untagged."""
    value = df.attrs.get(PARTITION_KEY)
    if value is None:
        return None
    try:
        return Partition(value)
    except ValueError:
        return None


def require_train(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the frame is the train partition; return it unchanged.

    Raises
    ------
    LeakageError
        If the frame is untagged (e.g. raw unsplit data) or tagged val/test —
        the loud, immediate failure that makes train-only fitting easy to
        honor and hard to violate silently.
    """
    role = partition_of(df)
    if role is None:
        raise LeakageError(
            "fit() called on an untagged frame. Only the train partition returned by "
            "split_data() may be fitted; transform() accepts any partition. "
            "This guard prevents fitting statistics on unsplit or evaluation data."
        )
    if role is not Partition.TRAIN:
        raise LeakageError(
            f"fit() called on the {role.value.upper()} partition. Statistics must be "
            "learned on train only; val/test must be transformed, never fitted."
        )
    return df


def require_partition(df: pd.DataFrame, role: Partition) -> pd.DataFrame:
    """Assert the frame carries exactly the expected partition tag."""
    actual = partition_of(df)
    if actual is not role:
        expected = role.value
        found = actual.value if actual is not None else "untagged"
        raise LeakageError(f"expected the {expected.upper()} partition, found {found.upper()}")
    return df
