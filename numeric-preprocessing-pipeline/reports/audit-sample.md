# Data Audit Report

- **Rows:** 5,100
- **Columns:** 22
- **Estimated in-memory size:** 1.0 MiB
- **Target column:** target
- **Duplicate rows (exact):** 200

## Column profile

| column | dtype | null % | unique | mean | std | min | q01 | q50 | q99 | max | skew | IQR outliers | z>3 | flags |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| __row_index__ | UInt32 | 0.00 | 5100 | — | — | — | — | — | — | — | — | — | — | — |
| id | Int64 | 0.00 | 5000 | 2497 | 1444 | 0 | 49.99 | 2500 | 4948 | 4999 | 0.0005557 | 0 | 0 | id-like |
| entity | String | 0.00 | 37 | — | — | — | — | — | — | — | — | — | — | — |
| timestamp | String | 0.00 | 5000 | — | — | — | — | — | — | — | — | — | — | — |
| x_normal | Float32 | 3.25 | 4792 | 2.736e+06 | 8.328e+07 | -1e+09 | -2.447 | 0.002496 | 2.835 | 1e+09 | 1.876 | 98 | 50 | leak?|r(target,x_normal)|=1.0000 |
| x_uniform | Float32 | 4.55 | 4773 | 0.00523 | 2.884 | -4.997 | -4.911 | 0.07596 | 4.897 | 5 | -0.01086 | 0 | 0 | — |
| x_exponential | Float32 | 7.98 | 4583 | — | — | — | 0.01154 | 1.395 | 10.12 | — | — | 245 | 0 | — |
| x_bimodal | Float32 | 10.10 | 4497 | 0.03848 | 2.065 | -3.594 | -3.006 | 0.9181 | 3.076 | 3.841 | -0.02356 | 0 | 0 | — |
| x_lognormal | Float32 | 15.20 | 4244 | 1.661 | 2.271 | 0.03275 | 0.09949 | 0.9966 | 10.41 | 59.62 | 7.604 | 339 | 74 | — |
| denominator | Float32 | 4.80 | 4765 | 9.946 | 3.043 | 0.1125 | 2.887 | 9.929 | 16.93 | 22.98 | 0.01937 | 34 | 11 | — |
| cat_low | String | 4.82 | 6 | — | — | — | — | — | — | — | — | — | — | near-constant |
| cat_bool | String | 9.94 | 3 | — | — | — | — | — | — | — | — | — | — | near-constant |
| const_col | Float32 | 0.00 | 1 | 42 | 0 | 42 | 42 | 42 | 42 | 42 | — | 0 | 0 | constant |
| near_const | Float32 | 0.00 | 2 | 1.036 | 0.4664 | 1 | 1 | 1 | 1 | 7 | 12.71 | 31 | 31 | near-constant |
| all_null_col | Float32 | 100.00 | 1 | — | — | — | — | — | — | — | — | 0 | 0 | constant |
| mostly_null | Float32 | 96.71 | 164 | 0.07244 | 0.9985 | -2.363 | -1.964 | 0.07076 | 2.256 | 2.592 | 0.05276 | 0 | 0 | — |
| x_corr_strong | Float32 | 3.71 | 4816 | -0.0226 | 0.9776 | -3.591 | -2.323 | -0.01407 | 2.302 | 3.477 | -0.003142 | 52 | 11 | — |
| x_corr_moderate | Float32 | 6.31 | 4686 | -0.01417 | 2.657 | -6.818 | -5.321 | 0.009422 | 5.136 | 7.256 | -0.02984 | 0 | 0 | — |
| numerator | Float32 | 6.73 | 4667 | 29.89 | 9.163 | -0.9126 | 8.584 | 29.78 | 50.88 | 67.61 | 0.01784 | 27 | 12 | — |
| x_mixed_format | Float32 | 0.00 | 4755 | 100 | 25.27 | 12.81 | 43.88 | 99.94 | 160.3 | 193.3 | 0.04851 | 30 | 14 | — |
| x_percent_string | Float32 | 0.00 | 288 | 14.92 | 4.972 | 0 | 3.4 | 14.9 | 26.7 | 37.5 | 0.03029 | 42 | 10 | — |
| target | Float32 | 1.96 | 4835 | — | — | — | -2.357 | 0.9085 | 4e+06 | — | — | 386 | 0 | — |

## Highly correlated pairs

| left | right | pearson r |
|---|---|---:|
| x_normal | target | 1.0000 |
| denominator | numerator | 0.9940 |
| x_uniform | x_corr_moderate | 0.9254 |

## Warnings

- column 'all_null_col' is entirely null

_The audit is informational: it never changes pipeline behavior. All fitted decisions are made on the train partition only._
