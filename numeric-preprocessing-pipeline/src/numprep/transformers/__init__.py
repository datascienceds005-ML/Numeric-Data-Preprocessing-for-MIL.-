"""Scikit-learn compatible, DataFrame-native transformers (all fitted on train only).

Every transformer in this package:

* subclasses :class:`numprep.transformers.base.FrameTransformer`
  (``BaseEstimator`` + ``TransformerMixin`` ⇒ ``get_params``/``set_params``,
  ``clone``-safety and ``set_output`` support),
* consumes and produces ``pandas.DataFrame`` with a preserved row index/order
  and a row-count assertion,
* implements ``get_feature_names_out``,
* stores fitted state exclusively in attributes suffixed ``_``,
* never sees the target column (the orchestrator pops it before the pipeline).
"""

from __future__ import annotations

from numprep.transformers.base import FrameTransformer
from numprep.transformers.encode import LowCardinalityEncoder
from numprep.transformers.features import FeatureEngineer
from numprep.transformers.impute import NumPrepImputer
from numprep.transformers.indicators import MissingIndicatorAdder
from numprep.transformers.outliers import OutlierClipper
from numprep.transformers.prune import TrainOnlyColumnPruner
from numprep.transformers.scale import GroupedScaler
from numprep.transformers.select import FeatureSelector

__all__ = [
    "FeatureEngineer",
    "FeatureSelector",
    "FrameTransformer",
    "GroupedScaler",
    "LowCardinalityEncoder",
    "MissingIndicatorAdder",
    "NumPrepImputer",
    "OutlierClipper",
    "TrainOnlyColumnPruner",
]
