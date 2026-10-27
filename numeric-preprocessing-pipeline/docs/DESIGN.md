# DESIGN — Numeric Data Preprocessing for ML (`numprep`)

**Status:** implemented as specified here. This document is the decision log; the code is
the source of truth for behavior, this document is the source of truth for *why*.

---

## 1. Problem & goals

Raw CSVs of messy numeric data must be audited, cleaned, engineered, split, and
serialized into train/validation/test partitions that are safe (no leakage), typed,
and ready for model development — reproducibly, in Docker, in CI, at sizes beyond RAM
for the ingest/audit stages.

**Goals**

- G1 — Correctness first: the split→fit→transform order is enforced *by construction*.
- G2 — Determinism: identical input + config + image ⇒ byte-identical Parquet outputs.
- G3 — One-command reproduction in Docker; drop-in CI gates (lint, types, tests,
  coverage ≥ 85 %, container smoke run, determinism gate).
- G4 — Every edge case in the brief has a named code path and a test.
- G5 — Engine-per-stage: Polars for scan/clean/audit, pandas/NumPy/sklearn for the
  fitted stage. No engine never leaves the stage boundary unmanaged.

**Non-goals**

- Model training, hyper-parameter search, feature stores, orchestration engines
  (Airflow/Prefect), distributed execution. DVC is recommended, not wired in.
- Non-numeric feature pipelines (text, images). Categorical handling is limited to
  low-cardinality one-hot encoding as explicitly requested.

---

## 2. Architecture

### 2.1 Stage diagram

```
                      ┌──────────────────────────────────────────────────────────┐
                      │                      orchestrator.run_pipeline           │
                      └──────────────────────────────────────────────────────────┘
 raw CSVs ──► [S1 ingest] ──► snapshot.parquet ──► [S2 validate] ──► [S3 audit] ──►
              polars lazy      zstd, typed         pandera +       report-only
              scan/stream      + __row_index__     native checks   JSON+MD
 ──► [S4 clean (stateless)] ──► [S5 split] ──► [S6 fit on TRAIN] ──► [S7 transform] ──►
      inf→NaN, domain rules,     random/         sklearn Pipeline       val/test via
      dedupe, null-target rows,  stratified/     (fit ONLY on the       fitted pipe;
      temporal lags/rollings     group/chrono    tagged TRAIN frame)    len preserved
 ──► [S8 output contracts] ──► [S9 outputs] ──► [S10 report]
      no NaN/inf, schema         train/val/test   reports/summary.md
      consistent across splits   .parquet(zstd)
                                 + preprocessor.joblib
                                 + fitted_stats.json
                                 + feature_catalog.json
                                 + manifest.json
```

### 2.2 Data flow & module boundaries

| Module | Responsibility | Engine | I/O |
|---|---|---|---|
| `config.py` | Typed Pydantic config, YAML + env + CLI overlay, cross-field validation | pydantic | reads YAML |
| `errors.py` | Exception hierarchy → exit codes | – | – |
| `jsonlog.py` | Structured JSON logging, stage timing + peak-RSS context manager | stdlib | stdout + `logs/run.jsonl` |
| `fsutils.py` | Atomic writes (tmp + `os.replace`), sha256, canonical JSON | stdlib/pyarrow | – |
| `seed.py` | Single global seed (`random`, `numpy`), `PYTHONHASHSEED` note | numpy | – |
| `ingest.py` | Path/glob resolution, encoding/BOM/duplicate-header handling, string→numeric coercion, dtype overrides, float32 downcast guard, snapshot write | polars | CSV → Parquet |
| `contracts.py` | Input contract (pandera pandas backend + native polars fast path), output contract (native, exact) | polars+pandera | – |
| `audit.py` | Profiling: nulls, quantiles, skew/kurtosis, constants, ID-like, dupes, inf, corr pairs, outliers, leakage flags — **report-only** | polars | JSON + MD |
| `clean.py` | Stateless: inf→NaN, domain rules, dedupe, null-target row drop | polars | in-memory |
| `temporal.py` | Lag / rolling features, strictly past-only, computed pre-split (see §4.3) | polars | in-memory |
| `partitions.py` | Partition tagging (`TRAIN/VAL/TEST` attrs), `require_train()` guard | pandas | – |
| `split.py` | random / stratified / group / chronological + fallbacks + overlap assertions | pandas/sklearn | – |
| `transformers/*` | 8 sklearn-compatible fitted transformers (see §5) | pandas/numpy | – |
| `pipeline.py` | Pipeline composition, `fit_preprocessor(train, …)` / `transform_partition(…)` API, joblib save/load | sklearn | joblib |
| `artifacts.py` | Manifest (deterministic/runtime blocks), fitted-stats extraction, feature catalog export | stdlib | JSON |
| `report.py` | Auto-populated `summary.md` from the actual run | stdlib | MD |
| `orchestrator.py` | Stage sequencing, stage logs, `RunResult` | – | – |
| `synthesize.py` | Seeded messy-data generator (all pathology classes) | numpy/pandas | CSV |
| `cli.py` | Typer app: `run`, `audit`, `validate`, `synthesize` | typer | exit codes |

Rules enforced in review: one responsibility per module; no circular imports
(dependencies point strictly downwards: cli → orchestrator → stages → utils);
no hidden global state (config is injected; the only module-level mutable state is
the logger, set up once in `cli.main`); pure-function cores with I/O at the edges
(`ingest`, `artifacts`, `report`, `cli`).

---

## 3. Configuration

Single YAML → Pydantic v2 models, `extra="forbid"` (unknown keys fail with the list
of allowed keys). Overlay order (lowest→highest): **pydantic defaults → YAML →
environment (`NUMPREP_SEED`, `NUMPREP_N_JOBS`) → CLI flags**. Zero-config runs are
possible (`numprep run --input data --output out`) — every field has a defensible
default. Cross-field rules (validated in `model_validator(mode="after")`):

- `split.strategy=chronological` ⇒ `split.time_column` required; `group` ⇒ `group_columns` required.
- `split.strategy=stratified` ⇒ `target.column` must be set.
- `cleaning.temporal.enabled=true` ⇒ strategy must be `chronological`.
- `input.thousands_separator` must differ from `input.delimiter`.
- Ratios are validated positive and renormalized to sum to exactly 1 (flag recorded
  in the run report when renormalization fired).

The full annotated schema lives in `configs/default.yaml` and README §5.

---

## 4. Leakage-prevention strategy (the core design constraint)

### 4.1 Structural rule

**Split first. Fit on train only. Transform val/test.** Enforced in three layers:

1. **Type-level nudge:** `split.py` returns pandas DataFrames whose `attrs` carry a
   partition tag (`TRAIN`/`VAL`/`TEST`), set only by the split module.
2. **Runtime guard:** every fitted transformer and `pipeline.fit_preprocessor` calls
   `partitions.require_train(df)`, which raises `LeakageError` (exit 5) unless the
   frame is tagged `TRAIN`. A raw/unsplit frame — or val/test — simply cannot be
   passed to `fit` without an explicit, loud failure. `transform` paths assert the
   frame is tagged and that the transformer is fitted (`check_is_fitted`).
3. **Row-count assertion:** every transform asserts `len(out) == len(in)` — rows are
   never silently dropped after the split (drops happen only in documented stateless
   stages before the split).

### 4.2 Subtle sources, handled explicitly

| Leakage source | Handling |
|---|---|
| Statistics computed pre-split (means, quantiles, corr, column drops) | All fitted behavior lives in sklearn transformers; the only pre-split computations are *stateless row ops* (inf→NaN, domain rules, dedupe, null-target drop, past-only temporal features). Nothing fitted touches pre-split data. |
| Column-drop decisions | Null-rate / zero-variance / ID-like drops are fitted on **train only** (`TrainOnlyColumnPruner`), even though dedupe is stateless-pre-split. Audit may *report* full-data facts but nothing consumes audit output for fitting (grep-enforced separation; test asserts pruner = train-recomputed, ≠ full-data). |
| Duplicates straddling splits | Exact duplicates are removed globally pre-split (stateless row op, no statistic), so identical rows cannot straddle. Near-duplicates: the split meta reports a rounded-value hash collision count across partitions (report-only); dropping near-duplicates is deliberately **not** automatic (see §8 tradeoffs). |
| Group/entity overlap | `group` strategy splits on group keys only; `SplitMeta` asserts zero key intersection across all three partitions (hard assertion, not a warning). |
| Time look-ahead | Chronological split is a contiguous ordered slice, never shuffled. Lag/rolling features use `shift(≥1)` before windowing (window ends at *t−1*), computed within entity groups when configured. Test proves `feat[t] == value[t−k]` and rolling mean at *t* excludes row *t*. |
| Target-derived columns | Audit flags `|corr(col, target)| > 0.95`, identity-with-target, and target-y name patterns — report-only. `pipeline.py` pops the target before fit and asserts it is absent from `X`. |
| Imputers/ selectors fitted with the target | Transformers never see `y`. There is no code path from `y` into any `fit`. |
| Audit driving fitted behavior | Audit output is written to reports; the orchestrator never passes audit results to any fitter (asserted by leakage test #6). |
| Scaler/imputer seeing val/test | `transform_partition` calls `pipe.transform` only; there is no `fit_transform` API exposed on the orchestrator path at all. |

### 4.3 Temporal features pre-split — why this is leak-free

Lag/rolling features are **stateless** (no parameters are estimated) and use **only
rows strictly earlier than the current row**. Computing them on the full
time-ordered frame before the split gives val/test rows their legitimate history
(standard practice in time-series ML) and cannot move future information backwards.
What would be leakage — estimating anything (e.g. a scaler) on the union — is
impossible here because these features carry no estimates, and the chronological
split is itself a strictly ordered slice. This reasoning is tested in
`tests/test_leakage.py::test_temporal_features_use_only_past`.

---

## 5. The stateful stage: one sklearn `Pipeline`

Chosen shape: a single `sklearn.pipeline.Pipeline` of **DataFrame-in/DataFrame-out**
transformers, each reading its column specification from config.

Steps, in order (names are the pipeline step keys):

1. `prune` — `TrainOnlyColumnPruner`: drops columns fitted on train whose null-rate
   exceeds `cleaning.null_column_threshold`, that are constant, or ID-like
   (name pattern or unique-ratio). Reasons recorded per column.
2. `indicators` — `MissingIndicatorAdder`: `f"{col}__was_missing"` flags for columns
   null in train (capped by `max_indicators`, chosen by null-count desc then name).
3. `impute` — `NumPrepImputer`: `median` (default) / `mean` / `constant` / `knn` /
   `iterative` (sklearn experimental; `random_state=seed`). Numeric columns only.
4. `clip` — `OutlierClipper`: `winsorize` (percentile) / `iqr` / `zscore` / `none`.
   Bounds fitted on train; transform clips in place (never drops rows) and logs
   clipped counts per partition.
5. `features` — `FeatureEngineer`: config-driven `log1p` (signed variant when the
   train min is negative), `ratios` (safe division: |denominator| < ε ⇒ NaN),
   `differences`, `interactions`, `polynomial`. Engineered NaN/±inf are replaced
   with train-medians of the engineered outputs. Emits the feature catalog
   (name, formula, rationale).
6. `encode` — `LowCardinalityEncoder`: one-hot for object/bool/category columns with
   ≤ `max_cardinality` train categories (sorted, deterministic); unseen categories
   at transform ⇒ all-zero row; high-cardinality string columns are dropped with a
   logged warning (fitted decision on train).
7. `select` — `FeatureSelector`: variance threshold, then greedy correlation pruning
   (drop the member of a high-|r| pair with the higher mean |r| to the rest;
   deterministic tie-breaks; seeded row sampling above `sample_size`).
8. `scale` — `GroupedScaler`: per-column scaler chosen by config override or
   `auto` skew rule (|skew| ≥ 1 ⇒ `power` Yeo-Johnson; ≥ 0.5 ⇒ `robust`; else
   `standard`; near-constant ⇒ passthrough) with reasons recorded. `^cat_` and
   `__was_missing$` columns are excluded by default.

Rejected alternative: a `ColumnTransformer` of column groups. With global ops
(winsorization, feature engineering, per-column auto-scaling) and dynamic column
addition/removal, a `ColumnTransformer` forces `get_feature_names_out` bookkeeping
on every hop for no gain — all heterogeneity here is *per-column scaler choice*,
which `GroupedScaler` already encapsulates. Tradeoff recorded, not hidden.

The fitted pipeline is serialized **once** with `joblib` (compress=3). Custom
transformers implement `get_params/set_params` (via `BaseEstimator`),
`get_feature_names_out`, `set_output` (via `TransformerMixin`), and are
`clone()`-safe (no mutable default arguments; fitted state only in `*_` attrs).

**Target column:** popped before the pipeline; never imputed, scaled, or used in any
fit; reattached unchanged for output. `output` dtype consistency is enforced at
write time.

---

## 6. Edge-case handling matrix (implemented + tested)

| Case | Behavior | Where |
|---|---|---|
| Empty file / header-only file | skipped with warning; all-empty ⇒ `InputContractError` (exit 3) | ingest |
| Inconsistent columns across files | diagonal union (missing ⇒ null), logged | ingest |
| Duplicate column names | deterministic `name__dupN` renaming | ingest |
| BOM | stripped | ingest |
| Non-UTF8 bytes | `encoding=utf-8-lossy` (default) or `latin-1` manual decode | ingest |
| Mixed-type / numeric strings (“1,234”, “ 42 ”, “(3.5)”, “$7”) | strip/clean then `strict=False` cast; junk→null counted; columns ≥ `numeric_min_castable_ratio` castable become numeric | ingest |
| Entirely-null column | kept as null column; audit reports; train-fitted pruner drops | ingest/audit/pipeline |
| Constant / near-constant columns | audit flags; pruner drops constants; scaler passes through near-constants | audit/pipeline |
| All files missing / no glob match | `InputContractError` exit 3 with matched-pattern message | ingest |
| Required column missing / range violated / null-rate ceiling | `InputContractError` exit 3, all violations aggregated | contracts |
| Ratios ≠ 1 | renormalized, flagged in report | config/split |
| Tiny datasets (n < 3) | `PipelineError` with guidance; n=3+ ⇒ ≥1 row per split guaranteed (largest-remainder sizing) | split |
| Rare classes (count < 3) | stratified ⇒ random fallback, warning + report note | split |
| Missing target values | rows dropped pre-split (configurable), count logged | clean |
| Fewer groups than splits (n_groups < 3) | group ⇒ random fallback with warning; retry loop over seeds before falling back | split |
| Groups fewer than needed / all-in-one-side | deterministic retry (seed+i, ≤100 attempts), then fallback | split |
| Null timestamps | rows dropped with count logged (chronological only) | split |
| Unseen categories in val/test | all-zero one-hot row | encode |
| New NaN/inf created by engineering | replaced with train-median of engineered feature | features |
| Extreme magnitudes breaking float32 | downcast guard keeps float64 and logs | ingest |
| Negative values with log features | signed log: `sign(x)·log1p(|x|)` | features |
| log1p of all-null column | skipped with warning | features |
| Zero-IQR columns under IQR clip | bounds collapse to the constant; no NaN created | outliers |

---

## 7. Performance strategy

- **Ingest/audit/clean:** Polars lazy scans; one streaming `collect` for the audit’s
  per-column aggregations (single pass); `engine="streaming"` available on collect;
  string-first ingestion with explicit casts (predictable memory, no schema
  surprises). Files are read sequentially and concatenated lazily; nothing requires
  the full raw CSV in RAM before the snapshot is written.
- **Correlations & selection:** seeded sampling above `sample_size` (default 200k
  rows); correlation-pair analysis capped at `audit.max_corr_columns` (default 100,
  skip with warning beyond).
- **Duplicate scan:** full-data exact scan up to `audit.max_duplicate_scan_rows`
  (default 2 M), else seeded 200 k-row sampled estimate, clearly labeled.
- **dtypes:** float32 downcast with a deterministic strided-sample precision guard
  (`rtol=1e-6`); integer semantics preserved where exact; `__row_index__` is UInt32.
- **sklearn stage:** pandas/NumPy at the fitted stage (sklearn compatibility);
  `n_jobs` forwarded where supported. Peak RSS per stage measured via
  `getrusage(ru_maxrss)` deltas — documented as a process high-water mark (monotonic;
  per-stage deltas under-report, honestly noted in the benchmark).
- **Known scaling limit (stated honestly):** the fitted sklearn stage operates
  in-memory. Datasets that fit in RAM for the *fitted* stage but not for *raw text
  ingest* are the supported envelope; beyond that, the right move is upstream
  sampling or a Spark/Polars-native reimplementation of the fitted stage (recorded
  in §8 rejected tradeoffs).

## 7.1 Determinism

- One seed (`run.seed`) drives `random`, `numpy`, sklearn `random_state`s, Hypothesis
  (derandomized), sampling, split shuffles.
- `PYTHONHASHSEED` is exported in the Docker image and documented: Python cannot
  retroactively fix its hash seed in-process, so the code **never relies on
  set/dict iteration order for decisions** (everything decision-bearing is sorted).
- Sorted/stable ordering everywhere: file lists, dedupe `keep="first"` with
  `maintain_order=True`, chronological tie-break on `__row_index__`, sorted
  categories, sorted manifest keys.
- No timestamps inside data files or the deterministic manifest block; Parquet
  metadata contains only library-derived constant strings (verified by the
  byte-identical-hash test across fresh processes).

---

## 8. Tradeoffs considered and rejected

1. **Great Expectations / heavyweight profilers** — GX brings a DX/data-docs stack
   out of proportion to this deliverable; pandera gives contract enforcement with a
   fraction of the surface. Rejected.
2. **pandera polars backend** — probed in this environment; container-level check
   failure reporting is buggy (crash inside `failure_cases_metadata`). Pandas
   backend chosen on a sampled frame for the input contract; native polars checks do
   the exact fast path. Re-evaluate on pandera ≥ 0.35.
3. **Dask / Spark** — unjustified operational weight for a single-node container
   deliverable; polars streaming covers the > RAM *ingest/audit* envelope. Rejected.
4. **Dropping outlier rows** — forbidden by the brief; winsorization/IQR/z-score
   caps with logged counts instead. Rows are never dropped after the split.
5. **Automatic near-duplicate removal across splits** — near-duplicate definition is
   data-dependent (tolerances); silently dropping val/test rows would corrupt split
   semantics. We *report* cross-partition rounded-hash collisions and leave the
   decision to the analyst. Rejected automatic removal.
6. **IterativeImputer as default** — powerful but slow and order-sensitive; median
   is the defensible default, iterative opt-in. Rejected as default.
7. **HTML-only human reports** — Markdown diffs well in PRs and renders on forges;
   JSON carries the machine-readable half. HTML export is a config flag away but
   not default. Rejected as default.
8. **ColumnTransformer composition** — see §5.
9. **Fitting temporal features post-split per partition** — would break val/test
   history continuity (first rows would have no lag source) for zero leakage
   benefit; see §4.3.

---

## 9. Assumption log (also mirrored in README)

- A1 — Default split is **random 0.70/0.15/0.15**; stratified requires a target;
  chronological requires `time_column`; all overridable.
- A2 — "Target" is a single column (regression or binary/multiclass classification).
  No multi-target support.
- A3 — `median` imputation, `winsorize(0.01, 0.99)` outlier handling, `auto` scaling
  are defensible industry defaults; each is configurable.
- A4 — ID-like detection = name regex *or* train unique-ratio ≥ 0.999; flagged in
  audit, dropped by the train-fitted pruner only when `cleaning.drop_id_like=true`.
- A5 — Near-constant threshold: `n_unique/rows < 0.01` with `n_unique ≤ 10` (report
  only); the pruner drops only exactly-constant columns by default.
- A6 — Classification vs regression auto-detection: non-float dtype, or float with
  all-integral values and ≤ 20 uniques ⇒ classification. Overridable via
  `target.task`.
- A7 — The audit is a *report*, not a controller; configs own every decision.
- A8 — `encoding_errors` semantics: `utf-8-lossy` ≈ Python `errors="replace"`.
- A9 — Git SHA is recorded when available (CI or a dev checkout); absent in minimal
  containers ⇒ `null`. This does not affect determinism of data outputs.
- A10 — Docker is unavailable in the delivery sandbox; the Dockerfile/CI/smoke
  script are fully written and locally dry-validated, and the smoke script no-ops
  with a clear message when `docker` is absent.

---

## 10. Exit codes

| Code | Meaning | Raised by |
|---|---|---|
| 0 | success | – |
| 1 | unexpected CLI wiring error (reserved; commands map everything below) | – |
| 2 | configuration error **or** CLI usage error (Click usage = 2 by convention) | `ConfigError`, argparse/click |
| 3 | input contract failure (missing files/columns, dtype/range/null-rate violations, unreadable input) | `InputContractError` |
| 4 | output contract failure (NaN/inf in outputs, inconsistent schema across splits, empty split) | `OutputContractError` |
| 5 | unexpected pipeline error (including `LeakageError`) | `PipelineError`, `LeakageError` |
| 130 | SIGINT | – |
