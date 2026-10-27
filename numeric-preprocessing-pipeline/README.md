# numprep — Numeric Data Preprocessing for ML

Production-grade, reproducible, containerized preprocessing pipeline for **messy
numeric datasets**: audit raw CSVs, clean, engineer features, split safely, and
export model-ready train/validation/test partitions — with **train-only
fitting enforced by construction** and byte-identical deterministic outputs.

```
raw CSVs ─► ingest (Polars streaming, typed Parquet snapshot)
         ─► input contract (pandera + native, exit 3 on violation)
         ─► audit (JSON + Markdown, report-only)
         ─► stateless clean (inf→null, domain rules, dedupe, temporal lags)
         ─► split (random / stratified / group / chronological)
         ─► FIT ON TRAIN ONLY (sklearn pipeline; LeakageError otherwise)
         ─► transform val/test ─► output contract (no NaN/inf, schema-locked)
         ─► Parquet (zstd) + preprocessor.joblib + fitted_stats + catalog
            + manifest.json + reports/summary.md
```

---

## 0. Quickstart in GitHub Codespaces (recommended)

The repository ships a `.devcontainer/` — Python 3.12 image, Docker-in-Docker,
VS Code extensions (Ruff, mypy, YAML, Actions, Docker), and a `postCreateCommand`
that installs the package with dev tooling before your first terminal opens.

1. Get the repository into GitHub (upload via the GitHub web UI, `git push` from
   anywhere, or `gh repo create` inside any environment with `gh` authenticated).
2. On the repo page: **Code → Codespaces → Create codespace on `main`**.
3. When the codespace is ready (post-create prints `Devcontainer ready`):

   ```bash
   make verify                        # ruff + mypy + pytest with the 85% gate
   make synthesize && make run-local  # seeded messy data → full pipeline
   ls outputs/run                     # parquet partitions, preprocessor.joblib,
                                      # manifest.json, fitted stats, summary report
   ```

Because Docker-in-Docker is enabled, `make build`, `make run`, and
`make docker-smoke` all work inside the codespace exactly as they do in CI.
Run `make help` to list every target.

## 1. Quickstart (one command)

```bash
docker build -t numprep:local .

mkdir -p data && cp /path/to/*.csv data/   # raw inputs
docker run --rm \
  -v "$PWD/data:/data:ro" \                 # read-only inputs
  -v "$PWD/out:/out" \                      # writable outputs
  numprep:local run --input /data --output /out
```

No config file needed — built-in defaults are sensible (median imputation,
winsorized outliers, skew-aware auto scaling, random 70/15/15 split). With a
target column add `-e NUMPREP_...` or a mounted config (§5).

Everything is reproducible: same input + config + image ⇒ **byte-identical
Parquet outputs** (verified in CI by running twice in fresh containers with
different `PYTHONHASHSEED` values).

## 2. Local development

```bash
make install-dev     # pip install -e ".[dev]"
make verify          # ALL fast gates: ruff + mypy + pytest w/ coverage
make lint            # ruff check + format check
make typecheck       # mypy (pydantic plugin, strict-ish)
make test            # pytest + coverage, 85% gate (suite: ~88%)
make run-local       # run on data/raw → outputs/run
make determinism     # two fresh processes, compare output hashes
```

Python **3.12** recommended (3.11+ works; exact pins in `requirements.txt`).

## 3. CLI reference

Entry points: `numprep …` (installed console script) or `python -m numprep …`.

| Command | Purpose | Notable flags |
|---|---|---|
| `run` | Full pipeline | `--config/-c`, `--input/-i` (repeatable), `--output/-o`, `--seed`, `--n-jobs`, `--log-level`, `--export-csv/--no-export-csv` |
| `audit` | Ingest + validate + audit only (snapshot + `audit/audit.{json,md}`) | `--config`, `--input`, `--output`, `--seed`, `--log-level` |
| `validate` | Config + input-contract validation; writes nothing | `--config`, `--input`, `--log-level` |
| `synthesize` | Seeded messy-dataset generator (tests/demos/smoke) | `--rows`, `--seed`, `--task regression\|classification`, `--out` |

Exit codes (also in `docs/DESIGN.md` §10):

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | configuration error **or** CLI usage error |
| 3 | input contract failure (missing files/columns, dtypes, ranges, null ceilings) |
| 4 | output contract failure (NaN/inf, schema drift across splits, empty split) |
| 5 | unexpected pipeline error (includes `LeakageError`) |
| 130 | SIGINT |

Environment overrides: `NUMPREP_SEED`, `NUMPREP_N_JOBS` (CLI flags win over
env, env wins over YAML, YAML wins over built-in defaults).

## 4. Output layout

```
<output-dir>/
├── snapshot/data.parquet        # typed, coerced full snapshot (input to all stages)
├── data/{train,val,test}.parquet (+ .csv when --export-csv)
│                                # columns: [features…, target, __row_index__]
├── artifacts/
│   ├── preprocessor.joblib      # fitted sklearn pipeline (reload & transform)
│   ├── fitted_stats.json        # human-readable: fills, bounds, scalers + reasons
│   ├── feature_catalog.json     # engineered features: name, formula, rationale
│   └── manifest.json            # deterministic block + runtime block (timestamps here only)
├── audit/audit.json, audit.md   # profiling: nulls, quantiles, dupes, corr, leak flags
├── reports/summary.md           # auto-generated decision report from THIS run
└── logs/run.jsonl               # structured JSON logs (stdout mirrors them)
```

Determinism contract: the manifest's `deterministic` block (config hash, input
hashes, library versions, split sizes, output hashes) is byte-identical across
runs with identical input+config+image; timestamps live only in the
`runtime` block.

## 5. Configuration

Full annotated schema: [`configs/default.yaml`](configs/default.yaml). Worked
examples: [`example_classification.yaml`](configs/example_classification.yaml)
(stratified), [`example_timeseries.yaml`](configs/example_timeseries.yaml)
(chronological + lag/rolling features), [`example_group.yaml`](configs/example_group.yaml)
(group-aware). Unknown keys are rejected with the offending name — typos fail
fast. Key decisions and their knobs:

- **Split** — `split.strategy`: `random` (default), `stratified` (requires
  target; falls back to random with a warning for rare classes), `group`
  (requires `group_columns`; zero-overlap asserted), `chronological`
  (requires `time_column`; contiguous slice, never shuffled). Ratios are
  renormalized and flagged when they don't sum to 1.
- **Column drops are train-fitted** — null ceiling (`cleaning.null_column_threshold`),
  constants, and ID-like columns (name pattern **or** integer/string
  unique-ratio) are dropped by a transformer fitted on train only. The audit
  reports the same phenomena on full data but never drives decisions.
- **Outliers** — bounds learned on train (`winsorize` default, `iqr`, `zscore`,
  `none`); transform clips in place, never drops rows, logs counts.
- **Feature engineering** — `log1p` (auto-signed for negatives), `ratios`
  (safe division → NaN → train median), `differences`, `interactions`,
  `polynomial`; every feature lands in the catalog with formula + rationale.
- **Scaling** — per-column with `auto`: \|skew\| ≥ 1 → Yeo-Johnson power,
  ≥ 0.5 → robust, else standard; near-constants pass through; every decision
  recorded with its reason in `fitted_stats.json`.
- **Temporal features** — past-only lags/rollings (`shift ≥ 1` before
  windowing), computed pre-split on the ordered frame so val/test rows keep
  their legitimate history without look-ahead (DESIGN §4.3).
- **Categoricals** — one-hot for ≤ `max_cardinality` train categories; unseen
  categories → all-zero rows; high-cardinality strings dropped with a warning.

## 6. Leakage prevention (why you can trust the splits)

- The **only** frames accepted by any `fit` are the tagged train partition —
  `fit_preprocessor(pipe, train, cfg)` raises `LeakageError` (exit 5) for
  val/test or untagged/unsplit data. Transform paths assert tags and preserve
  row counts exactly.
- Column-drop / imputation / selection / scaling statistics are all learned on
  train; the target is popped before the pipeline and can never be imputed,
  scaled, or fitted on.
- Exact duplicates are removed pre-split (stateless), so identical rows cannot
  straddle partitions; near-duplicate collisions *across* partitions are
  reported (rounded-hash) but deliberately not auto-dropped (DESIGN §8.5).
- Automated proof: `tests/test_leakage.py` — train-only statistics, corrupted
  val/test leave the fitted artifact byte-identical, zero group overlap,
  past-only temporal features, audit-decoupling.

## 7. Testing

```bash
make test-fast     # 160+ tests: units, edge cases, leakage, determinism, e2e
```

Highlights: Hypothesis property tests (derandomized for CI), determinism tests
across fresh processes with different hash seeds, a joblib round-trip test
(reloaded preprocessor reproduces the stored test partition exactly), and
contract tests asserting exact exit codes via real subprocesses.

## 8. Benchmark & honest limits

```bash
make benchmark     # writes reports/benchmark.md
```

- Ingest/audit/clean use Polars streaming with sampled approximations
  (correlations, duplicate scan) — handles data beyond RAM for text ingest.
- The fitted sklearn stage is in-memory: the dataset must fit in RAM *after*
  the string→numeric snapshot conversion. Beyond that, sample upstream or move
  the fitted stage to a distributed engine (DESIGN §7, §8).
- Peak-RSS instrumentation uses `getrusage(ru_maxrss)` — a monotonic
  process-level high-water mark; per-stage deltas are lower bounds (documented
  in the benchmark output itself).

## 9. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| exit 3 “no files matched pattern” | `input.paths` glob wrong relative to cwd; use `--input` |
| exit 3 “null rate … exceeds ceiling” | contract is stricter than the data — raise `validation.input.max_null_rate` for that column or `{"*": …}` |
| exit 4 “feature column … non-numeric” | a string column survived: enable `preprocessing.encoding`, raise `max_cardinality`, or drop the column |
| exit 4 NaN in outputs | an all-null column can't be imputed: lower `null_column_threshold` so it's dropped, or fix upstream |
| `LeakageError` | you called `fit`/`fit_preprocessor` on val/test or untagged data — fit on `split_data(...).train` only |
| stratified “fell back to random” | a class has < 3 members; collect more or use `group` |
| nondeterministic outputs? | rerun with `make determinism`; if it fails, check for changed library versions (manifest pins them) |

## 10. Assumptions & documented decisions

All architectural decisions, rejected tradeoffs, and the assumption log live in
[`docs/DESIGN.md`](docs/DESIGN.md). The run's own `reports/summary.md` lists
the assumptions that applied to *that* run. A sample generated from a real run
is committed at [`reports/summary.md`](reports/summary.md).
