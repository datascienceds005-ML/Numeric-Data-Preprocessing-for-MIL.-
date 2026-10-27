"""Typed configuration system: YAML file + environment + CLI overlay, validated by Pydantic v2.

Overlay order (lowest to highest precedence):

1. Pydantic field defaults (the pipeline runs with zero configuration),
2. the YAML file passed via ``--config``,
3. environment variables ``NUMPREP_SEED`` and ``NUMPREP_N_JOBS``,
4. CLI flags (``--seed``, ``--input``, ``--output``, ``--n-jobs``, ``--export-csv``).

All models forbid extra keys, so a typo in YAML fails with an actionable error that
names the offending key. Cross-field consistency rules are enforced in
``NumPrepConfig``'s model validator (e.g. a chronological split requires a time
column). Every error surfaces as :class:`~numprep.errors.ConfigError` (exit code 2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from numprep.errors import ConfigError

# ---------------------------------------------------------------------------
# Type aliases (kept as Literals so YAML values are validated, not just typed)
# ---------------------------------------------------------------------------

SplitStrategy = Literal["random", "stratified", "group", "chronological"]
TaskType = Literal["auto", "classification", "regression"]
ImputationStrategy = Literal["median", "mean", "constant", "knn", "iterative"]
OutlierMethod = Literal["winsorize", "iqr", "zscore", "none"]
ScalerKind = Literal["auto", "standard", "robust", "minmax", "power", "quantile", "none"]
DowncastMode = Literal["auto", "force", "off"]
EncodingChoice = Literal["utf-8", "utf-8-lossy", "latin-1"]
DTypeOverride = Literal["float32", "float64", "int64", "int32", "bool", "string"]
RollingStat = Literal["mean", "std", "min", "max"]

SPLIT_KEYS = frozenset({"train", "val", "test"})
_DEFAULT_ROLLING: list[RollingStat] = ["mean"]


class _Strict(BaseModel):
    """Base model: unknown keys are rejected with an actionable error."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Run-level and input sections
# ---------------------------------------------------------------------------


class RunConfig(_Strict):
    """Global run settings."""

    seed: int = 42
    n_jobs: int = -1
    downcast: DowncastMode = "auto"


class InputConfig(_Strict):
    """Raw CSV input handling."""

    paths: list[str] = Field(default_factory=lambda: ["data/raw/*.csv"])
    delimiter: str = ","
    encoding: EncodingChoice = "utf-8-lossy"
    has_header: bool = True
    na_tokens: list[str] = Field(
        default_factory=lambda: [
            "",
            "NA",
            "N/A",
            "null",
            "NULL",
            "NaN",
            "nan",
            "None",
            "?",
            "-",
            "--",
        ]
    )
    dtypes: dict[str, DTypeOverride] = Field(default_factory=dict)
    thousands_separator: str | None = None
    currency_symbols: list[str] = Field(default_factory=lambda: ["$", "%", "€", "£"])
    parentheses_negative: bool = True
    numeric_min_castable_ratio: float = Field(0.5, ge=0.0, le=1.0)
    snapshot: bool = True

    @model_validator(mode="after")
    def _check_delimiter(self) -> InputConfig:
        if len(self.delimiter) != 1:
            raise ValueError(f"input.delimiter must be a single character, got {self.delimiter!r}")
        if self.thousands_separator is not None and self.thousands_separator == self.delimiter:
            raise ValueError(
                f"input.thousands_separator ({self.thousands_separator!r}) must differ from "
                f"input.delimiter ({self.delimiter!r}) or numeric parsing becomes ambiguous"
            )
        return self


class TargetConfig(_Strict):
    """Target column semantics."""

    column: str | None = None
    task: TaskType = "auto"
    drop_null_rows: bool = True


class SplitConfig(_Strict):
    """Split strategy, ratios, and grouping/time keys."""

    strategy: SplitStrategy = "random"
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    time_column: str | None = None
    time_format: str | None = None
    group_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_ratios(self) -> SplitConfig:
        if set(self.ratios) != SPLIT_KEYS:
            raise ValueError(
                f"split.ratios must contain exactly the keys {sorted(SPLIT_KEYS)}, "
                f"got {sorted(self.ratios)}"
            )
        for key, value in self.ratios.items():
            if not (0.0 < value <= 1.0):
                raise ValueError(f"split.ratios[{key!r}] must be in (0, 1], got {value}")
        if sum(self.ratios.values()) <= 0:
            raise ValueError("split.ratios must sum to a positive number")
        return self


class TemporalConfig(_Strict):
    """Past-only lag/rolling features for chronological data (stateless, pre-split)."""

    enabled: bool = False
    columns: list[str] = Field(default_factory=list)  # empty = all numeric features
    lags: list[int] = Field(default_factory=list)
    rolling_windows: list[int] = Field(default_factory=list)
    rolling_stats: list[RollingStat] = Field(default_factory=lambda: list(_DEFAULT_ROLLING))
    group_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_positive(self) -> TemporalConfig:
        for lag in self.lags:
            if lag < 1:
                raise ValueError(f"cleaning.temporal.lags must be >= 1, got {lag}")
        for window in self.rolling_windows:
            if window < 1:
                raise ValueError(f"cleaning.temporal.rolling_windows must be >= 1, got {window}")
        if not self.rolling_stats and self.rolling_windows:
            raise ValueError(
                "cleaning.temporal.rolling_stats must be non-empty when windows are given"
            )
        return self


# ---------------------------------------------------------------------------
# Cleaning section (stateless pre-split ops + train-fitted drop rules)
# ---------------------------------------------------------------------------


class CleaningConfig(_Strict):
    """Stateless cleaning and train-fitted column-drop policy.

    Note: null-threshold / constant / ID-like column drops are *fitted decisions*
    implemented by ``TrainOnlyColumnPruner`` (fit on train only) — see DESIGN §4.2.
    """

    dedupe: bool = True
    dedupe_subset: list[str] = Field(default_factory=list)
    inf_to_nan: bool = True
    non_negative_columns: list[str] = Field(default_factory=list)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    null_column_threshold: float = Field(0.95, ge=0.0, le=1.0)
    drop_constant_columns: bool = True
    drop_id_like: bool = True
    id_name_pattern: str = r"(?i)^(id|.*[_-]id|uuid|guid|index|row_?num|.*[_-]num)$"
    id_unique_ratio: float = Field(0.999, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Preprocessing section (all fitted on train only)
# ---------------------------------------------------------------------------


class ImputationConfig(_Strict):
    """Feature imputation (numeric columns; target is never imputed)."""

    strategy: ImputationStrategy = "median"
    constant_value: float = 0.0
    knn_neighbors: int = Field(5, ge=1)
    iterative_max_iter: int = Field(10, ge=1)
    add_missing_indicators: bool = True
    max_indicators: int = Field(50, ge=0)


class OutliersConfig(_Strict):
    """Train-fitted outlier bounds; transform clips, never drops rows."""

    method: OutlierMethod = "winsorize"
    lower_quantile: float = Field(0.01, ge=0.0, le=1.0)
    upper_quantile: float = Field(0.99, ge=0.0, le=1.0)
    iqr_k: float = Field(1.5, gt=0.0)
    zscore_cap: float = Field(3.0, gt=0.0)

    @model_validator(mode="after")
    def _check_quantiles(self) -> OutliersConfig:
        if self.method == "winsorize" and self.lower_quantile > self.upper_quantile:
            raise ValueError(
                f"preprocessing.outliers.lower_quantile ({self.lower_quantile}) must be "
                f"<= upper_quantile ({self.upper_quantile})"
            )
        return self


class RatioSpec(_Strict):
    """``name = numerator / denominator`` with safe division."""

    numerator: str
    denominator: str
    name: str | None = None


class DiffSpec(_Strict):
    """``name = minuend - subtrahend``."""

    minuend: str
    subtrahend: str
    name: str | None = None


class InteractionSpec(_Strict):
    """``name = left * right``."""

    left: str
    right: str
    name: str | None = None


class PolySpec(_Strict):
    """``name = column ** degree``."""

    column: str
    degree: int = Field(2, ge=2, le=5)
    name: str | None = None


class FeaturesConfig(_Strict):
    """Config-driven feature engineering with a generated catalog."""

    log1p: list[str] = Field(default_factory=list)
    ratios: list[RatioSpec] = Field(default_factory=list)
    differences: list[DiffSpec] = Field(default_factory=list)
    interactions: list[InteractionSpec] = Field(default_factory=list)
    polynomial: list[PolySpec] = Field(default_factory=list)
    safe_division_eps: float = Field(1e-12, ge=0.0)
    max_engineered: int = Field(200, ge=0)


class SelectionConfig(_Strict):
    """Train-fitted feature selection."""

    variance_threshold: float = Field(0.0, ge=0.0)
    correlation_threshold: float = Field(0.98, ge=0.0, le=1.0)
    sample_size: int = Field(200_000, gt=0)


class ScalingConfig(_Strict):
    """Per-column scaling with skew-aware ``auto`` selection."""

    default: ScalerKind = "auto"
    overrides: dict[str, ScalerKind] = Field(default_factory=dict)
    exclude_patterns: list[str] = Field(default_factory=lambda: [r"^cat_", r"__was_missing$"])
    skew_high: float = Field(1.0, gt=0.0)
    skew_moderate: float = Field(0.5, gt=0.0)
    quantile_output_distribution: Literal["normal", "uniform"] = "normal"
    quantile_n_quantiles: int = Field(1000, ge=10)

    @model_validator(mode="after")
    def _check_skew_thresholds(self) -> ScalingConfig:
        if self.skew_moderate > self.skew_high:
            raise ValueError(
                f"preprocessing.scaling.skew_moderate ({self.skew_moderate}) must be "
                f"<= skew_high ({self.skew_high})"
            )
        return self


class EncodingConfig(_Strict):
    """One-hot encoding of low-cardinality categorical/boolean columns."""

    enabled: bool = True
    max_cardinality: int = Field(10, ge=1)
    onehot_max_categories: int = Field(25, ge=1)
    drop_high_cardinality: bool = True


class PreprocessingConfig(_Strict):
    """Container for all train-fitted transformations."""

    imputation: ImputationConfig = Field(default_factory=ImputationConfig)
    outliers: OutliersConfig = Field(default_factory=OutliersConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)


# ---------------------------------------------------------------------------
# Audit section
# ---------------------------------------------------------------------------


class AuditConfig(_Strict):
    """Audit/profiling knobs (report-only: audit never drives fitted behavior)."""

    sample_size: int = Field(200_000, gt=0)
    max_corr_columns: int = Field(100, ge=2)
    max_duplicate_scan_rows: int = Field(2_000_000, gt=0)
    corr_threshold: float = Field(0.9, ge=0.0, le=1.0)
    leakage_corr_threshold: float = Field(0.95, ge=0.0, le=1.0)
    leakage_name_pattern: str = r"(?i)(^|_)(target|label|outcome|leak)"


# ---------------------------------------------------------------------------
# Validation contracts
# ---------------------------------------------------------------------------


class RangeSpec(_Strict):
    """Inclusive value range for a column; either bound may be omitted."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> RangeSpec:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"range min ({self.min}) must be <= max ({self.max})")
        return self


class InputContractConfig(_Strict):
    """Input data contract enforced before any processing."""

    enabled: bool = True
    required_columns: list[str] = Field(default_factory=list)
    expected_numeric: list[str] = Field(default_factory=list)
    ranges: dict[str, RangeSpec] = Field(default_factory=dict)
    max_null_rate: dict[str, float] = Field(default_factory=dict)
    allow_extra_columns: bool = True
    sample_rows: int = Field(1_000_000, gt=0)

    @model_validator(mode="after")
    def _check_null_rates(self) -> InputContractConfig:
        for column, ceiling in self.max_null_rate.items():
            if not (0.0 <= ceiling <= 1.0):
                raise ValueError(
                    f"validation.input.max_null_rate[{column!r}] must be in [0, 1], got {ceiling}"
                )
        return self


class OutputContractConfig(_Strict):
    """Output partition contract enforced after transformation."""

    forbid_nan_inf: bool = True
    require_nonempty_splits: bool = True
    consistent_schema: bool = True
    require_numeric_features: bool = True


class ValidationConfig(_Strict):
    """Input and output contracts."""

    input: InputContractConfig = Field(default_factory=InputContractConfig)
    output: OutputContractConfig = Field(default_factory=OutputContractConfig)


# ---------------------------------------------------------------------------
# Output section
# ---------------------------------------------------------------------------


class OutputConfig(_Strict):
    """Output locations and serialization options."""

    directory: Path = Path("outputs/run")
    compression: Literal["zstd", "snappy", "gzip"] = "zstd"
    export_csv: bool = False


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class NumPrepConfig(_Strict):
    """Root configuration object validated as one unit."""

    run: RunConfig = Field(default_factory=RunConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> NumPrepConfig:
        if self.split.strategy == "chronological" and not self.split.time_column:
            raise ValueError("split.strategy='chronological' requires split.time_column to be set")
        if self.split.strategy == "group" and not self.split.group_columns:
            raise ValueError("split.strategy='group' requires split.group_columns to be set")
        if self.split.strategy == "stratified" and not self.target.column:
            raise ValueError("split.strategy='stratified' requires target.column to be set")
        if self.cleaning.temporal.enabled and self.split.strategy != "chronological":
            raise ValueError(
                "cleaning.temporal.enabled=true is only valid with split.strategy='chronological'"
            )
        if (
            self.cleaning.temporal.enabled
            and not self.cleaning.temporal.lags
            and not self.cleaning.temporal.rolling_windows
        ):
            raise ValueError(
                "cleaning.temporal.enabled=true requires at least one lag or rolling window"
            )
        return self


# ---------------------------------------------------------------------------
# Loading & overlays
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliOverrides:
    """Explicit CLI-provided overrides applied on top of YAML and env values."""

    input_paths: tuple[str, ...] | None = None
    output_dir: str | None = None
    seed: int | None = None
    n_jobs: int | None = None
    export_csv: bool | None = None


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Fold ``NUMPREP_SEED`` / ``NUMPREP_N_JOBS`` into the raw config dict."""
    env_seed = os.environ.get("NUMPREP_SEED")
    if env_seed is not None:
        try:
            data.setdefault("run", {})["seed"] = int(env_seed)
        except ValueError as exc:
            raise ConfigError(f"NUMPREP_SEED must be an integer, got {env_seed!r}") from exc
    env_jobs = os.environ.get("NUMPREP_N_JOBS")
    if env_jobs is not None:
        try:
            data.setdefault("run", {})["n_jobs"] = int(env_jobs)
        except ValueError as exc:
            raise ConfigError(f"NUMPREP_N_JOBS must be an integer, got {env_jobs!r}") from exc


def _apply_cli_overrides(data: dict[str, Any], overrides: CliOverrides) -> None:
    """Fold CLI flags into the raw config dict (highest precedence)."""
    if overrides.input_paths is not None:
        data.setdefault("input", {})["paths"] = list(overrides.input_paths)
    if overrides.output_dir is not None:
        data.setdefault("output", {})["directory"] = overrides.output_dir
    if overrides.export_csv is not None:
        data.setdefault("output", {})["export_csv"] = overrides.export_csv
    if overrides.seed is not None:
        data.setdefault("run", {})["seed"] = overrides.seed
    if overrides.n_jobs is not None:
        data.setdefault("run", {})["n_jobs"] = overrides.n_jobs


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as a compact, actionable message."""
    lines = ["configuration validation failed:"]
    for error in exc.errors()[:20]:
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        message = error.get("msg", "invalid value")
        value = repr(error.get("input"))[:120]
        lines.append(f"  - {location}: {message} (got {value})")
    if len(exc.errors()) > 20:
        lines.append(f"  ... and {len(exc.errors()) - 20} more errors")
    lines.append("Hint: unknown keys are rejected; see configs/default.yaml for the full schema.")
    return "\n".join(lines)


def load_config(path: Path | None = None, overrides: CliOverrides | None = None) -> NumPrepConfig:
    """Load, overlay, and validate configuration.

    Parameters
    ----------
    path:
        Optional YAML file. ``None`` means "defaults only".
    overrides:
        CLI overrides applied last.

    Returns
    -------
    NumPrepConfig
        Fully validated configuration.

    Raises
    ------
    ConfigError
        If the file is missing/unreadable, YAML is malformed, env values are
        invalid, or validation fails (including unknown keys).
    """
    data: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ConfigError(f"config file {path} is not valid UTF-8: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config root must be a mapping, got {type(loaded).__name__}")
        data = loaded

    _apply_env_overrides(data)
    if overrides is not None:
        _apply_cli_overrides(data, overrides)

    try:
        return NumPrepConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def config_from_dict(data: dict[str, Any]) -> NumPrepConfig:
    """Validate a raw config dict, wrapping Pydantic errors as :class:`ConfigError`.

    Library consumers and tests use this instead of ``model_validate`` directly
    so the error contract (exit code 2) is uniform.
    """
    try:
        return NumPrepConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


def config_fingerprint(cfg: NumPrepConfig) -> str:
    """Deterministic JSON dump of the config (used for the manifest config hash).

    ``output.directory`` is excluded on purpose: it is where results are
    *placed*, not how data is *transformed*, so two runs into different
    directories must produce the same config fingerprint (and therefore the
    same deterministic manifest block for the CI determinism gate).
    """
    from numprep.fsutils import canonical_json

    payload = cfg.model_dump(mode="json")
    if isinstance(payload.get("output"), dict):
        payload["output"].pop("directory", None)
    return canonical_json(payload)
