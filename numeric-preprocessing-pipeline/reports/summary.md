# Preprocessing Run Summary

*Run window (UTC):* 2026-10-18T07:59:18Z → 2026-10-18T07:59:19Z · *Seed:* 42

## 1. Inputs & snapshot

| input file | rows | sha256 |
|---|---|---|
| synthetic_messy.csv | 5100 | b30970967e83… |

Ingested snapshot: **5,100 rows × 22 columns** → `data.parquet`

| column | coercion | dtype | unresolved→null |
|---|---|---|---|
| id | integer_cast | Int64 | 0 |
| x_normal | float_cast | Float64 | 0 |
| x_uniform | float_cast | Float64 | 0 |
| x_exponential | float_cast | Float64 | 0 |
| x_bimodal | float_cast | Float64 | 0 |
| x_lognormal | float_cast | Float64 | 0 |
| denominator | float_cast | Float64 | 0 |
| const_col | float_cast | Float64 | 0 |
| near_const | float_cast | Float64 | 0 |
| mostly_null | float_cast | Float64 | 0 |
| x_corr_strong | float_cast | Float64 | 0 |
| x_corr_moderate | float_cast | Float64 | 0 |
| numerator | float_cast | Float64 | 0 |
| x_mixed_format | float_cast | Float64 | 0 |
| x_percent_string | float_cast | Float64 | 0 |
| target | float_cast | Float64 | 0 |

## 2. Data-quality findings

| column | null % | unique | skew | flags |
|---|---|---|---|---|
| all_null_col | 100.00 | 1 | — | constant |
| mostly_null | 96.71 | 164 | 0.053 | — |
| x_lognormal | 15.20 | 4244 | 7.604 | — |
| x_bimodal | 10.10 | 4497 | -0.024 | — |
| cat_bool | 9.94 | 3 | — | near-constant |
| x_exponential | 7.98 | 4583 | — | — |
| numerator | 6.73 | 4667 | 0.018 | — |
| x_corr_moderate | 6.31 | 4686 | -0.030 | — |
| cat_low | 4.82 | 6 | — | near-constant |
| denominator | 4.80 | 4765 | 0.019 | — |

- Duplicate rows removed: **100**
- ±inf values nulled: **48**
- Rows with null target dropped: **123**
| high-corr pair | r |
|---|---|
| x_normal ~ target | 1.0000 |
| denominator ~ numerator | 0.9940 |
| x_uniform ~ x_corr_moderate | 0.9254 |

| potential leakage column | reasons |
|---|---|
| x_normal | |r(target,x_normal)|=1.0000 |


## 3. Split

- Strategy: **random (stratified fallback)** · sizes {'train': 3414, 'val': 732, 'test': 731} · applied ratios {'train': 0.7, 'val': 0.15, 'test': 0.15}
  - Fallback: rare class with 1 member(s) cannot be stratified

## 4. Fitted decisions (train only)

| dropped column | reason (train-fitted) |
|---|---|
| id | id-like: name matches '(?i)^(id|.*[_-]id|uuid|guid|index|row_?num|.*[_-]num)$' or train unique ratio >= 0.999 |
| timestamp | id-like: name matches '(?i)^(id|.*[_-]id|uuid|guid|index|row_?num|.*[_-]num)$' or train unique ratio >= 0.999 |
| const_col | constant on train (zero variance) |
| all_null_col | train null rate 1.0000 > threshold 0.95 |
| mostly_null | train null rate 0.9672 > threshold 0.95 |

- Imputation: **median** on 23 numeric column(s); 11 missing-indicator column(s) added.
- Outliers: **winsorize** bounds fitted for 23 column(s); clipping counts logged per partition.
_none_

- Encoding: one-hot applied to 2 column(s); 1 dropped (high cardinality).
- Selection: 1 column(s) dropped.
| column | scaler | reason |
|---|---|---|
| x_normal | standard | auto: |skew|=0.104 < skew_moderate 0.5 → standard |
| x_uniform | standard | auto: |skew|=0.002 < skew_moderate 0.5 → standard |
| x_exponential | power | auto: |skew|=1.667 >= skew_high 1 → Yeo-Johnson |
| x_bimodal | robust | config override (preprocessing.scaling.overrides) |
| x_lognormal | power | auto: |skew|=2.650 >= skew_high 1 → Yeo-Johnson |
| denominator | standard | auto: |skew|=0.001 < skew_moderate 0.5 → standard |
| cat_cat_low__alpha | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_low__beta | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_low__delta | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_low__gamma | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_low__omega | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_bool__False | none | excluded by scaling.exclude_patterns '^cat_' |
| cat_cat_bool__True | none | excluded by scaling.exclude_patterns '^cat_' |
| x_corr_strong | standard | auto: |skew|=0.012 < skew_moderate 0.5 → standard |
| x_corr_moderate | standard | auto: |skew|=0.012 < skew_moderate 0.5 → standard |
| numerator | standard | auto: |skew|=0.006 < skew_moderate 0.5 → standard |
| x_mixed_format | standard | auto: |skew|=0.050 < skew_moderate 0.5 → standard |
| x_percent_string | standard | auto: |skew|=0.036 < skew_moderate 0.5 → standard |
| x_lognormal__was_missing | none | excluded by scaling.exclude_patterns '__was_missing$' |
| x_bimodal__was_missing | none | excluded by scaling.exclude_patterns '__was_missing$' |
| cat_bool__was_missing | none | excluded by scaling.exclude_patterns '^cat_' |
| x_exponential__was_missing | none | excluded by scaling.exclude_patterns '__was_missing$' |
| numerator__was_missing | none | excluded by scaling.exclude_patterns '__was_missing$' |
| x_corr_moderate__was_missing | none | excluded by scaling.exclude_patterns '__was_missing$' |
| cat_low__was_missing | none | excluded by scaling.exclude_patterns '^cat_' |


## 5. Assumptions

- Split-first discipline: all fitted statistics were learned on the train partition only; val/test were transformed, never fitted.
- The audit is informational — no fitted decision consumes audit output.
- Exact duplicate rows are removed pre-split (stateless, prevents cross-split straddling); near-duplicates are reported, not removed.
- Column-drop decisions (null ceiling, constants, ID-like) were fitted on train statistics only.
- Outlier handling clips values to train-fitted bounds; rows are never dropped after the split.
- Engineered features with non-finite values are replaced with train medians of the engineered outputs.
- Unseen categories in val/test map to all-zero one-hot rows.
- The target column was never imputed, scaled, or used in any fit.

## 6. Recommended next steps

- Columns with >30% nulls (all_null_col, mostly_null): review upstream collection; heavy imputation degrades signal quality.
- Potential target-leakage columns were flagged in the audit — inspect them before modeling; consider dropping post-outcome fields at the source.
- The random (stratified fallback) split fell back to random (rare class with 1 member(s) cannot be stratified); collect more minority samples/groups for stratification, or use group-aware evaluation.
- Yeo-Johnson was applied to 2 skewed column(s); if these are count/monetary features, consider transforming at the source instead.
- 1 feature(s) were correlation-pruned; if interpretability matters, prefer domain-driven retention over automatic pruning.
- 100 duplicate rows were removed; if duplicates recur between runs, add upstream uniqueness constraints.
- Validate downstream model quality with cross-validation on the train split only; treat val/test as untouched until final reporting.
- Version the raw inputs and this run's manifest with DVC (or an equivalent data registry) so every model can be traced to exact bytes.

## 7. Reproduction

```bash
# deterministic rerun (same input + config + image ⇒ byte-identical outputs)
make run   # or: numprep run --config <cfg> --input <dir> --output <dir>
```

| output | sha256 |
|---|---|
| train | 29026f3eafdbc0230e0a37034da21ffdf2ef4d62344f966209400023d0fe7965 |
| val | 85e929cd6dbfddf31034ffb216d13da13968b533d9050e9a737e8e907ee02c68 |
| test | 44a94f97e6e46993196a1c77fb3073eff871c7147d249d9f0ec4f56e387e17a3 |

Manifest SHA-256: `77b81eb6b7951523c721c11eb323089e9fc65e9bf1bc64105ff37af3acbc0dc7`

Library context: python 3.12.15, numprep 1.0.0, numpy 2.1.3, pandas 2.2.3, polars 2.0.0, pyarrow 26.0.0, scikit-learn 1.5.2, pydantic 2.12.5
