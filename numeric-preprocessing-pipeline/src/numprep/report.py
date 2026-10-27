"""Auto-generated decision report (``reports/summary.md``).

Everything in the report comes from the *actual run*: stage counters, split
metadata, fitted statistics, contract results. The template is implemented in
code (this module) — regenerating the report after a config change is just a
rerun. Sections: run metadata, data-quality findings, decisions & rationale,
assumptions, next-step recommendations, reproduction command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from numprep.artifacts import library_versions
from numprep.audit import AuditResult
from numprep.clean import CleanStats
from numprep.ingest import IngestResult
from numprep.split import SplitMeta


@dataclass(frozen=True)
class RunContext:
    """Aggregated inputs for the report writer."""

    seed: int
    started_at_utc: str
    finished_at_utc: str
    ingest: IngestResult
    audit: AuditResult
    clean: CleanStats
    split_meta: SplitMeta
    fitted_stats: dict[str, Any]
    output_files: dict[str, str]
    contract_input: dict[str, Any]
    contract_output: dict[str, Any]
    manifest_sha256: str | None
    config_summary: dict[str, Any]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a Markdown table (empty rows → a dash line)."""
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def _recommendations(ctx: RunContext) -> list[str]:
    """Rule-based next-step suggestions derived from the run's actual findings."""
    recommendations: list[str] = []
    high_null = [c.name for c in ctx.audit.per_column if c.role != "index" and c.null_rate > 0.3]
    if high_null:
        recommendations.append(
            f"Columns with >30% nulls ({', '.join(high_null[:5])}): review upstream "
            "collection; heavy imputation degrades signal quality."
        )
    if ctx.audit.leakage_flags:
        recommendations.append(
            "Potential target-leakage columns were flagged in the audit — inspect them "
            "before modeling; consider dropping post-outcome fields at the source."
        )
    if ctx.split_meta.fallback:
        recommendations.append(
            f"The {ctx.split_meta.strategy} split fell back to random "
            f"({ctx.split_meta.fallback}); collect more minority samples/groups for "
            "stratification, or use group-aware evaluation."
        )
    scaling = ctx.fitted_stats.get("scaling", {})
    power_columns = [name for name, s in scaling.items() if s.get("kind") == "power"]
    if power_columns:
        recommendations.append(
            f"Yeo-Johnson was applied to {len(power_columns)} skewed column(s); if these "
            "are count/monetary features, consider transforming at the source instead."
        )
    selection = ctx.fitted_stats.get("selection", {}).get("dropped", {})
    if selection:
        recommendations.append(
            f"{len(selection)} feature(s) were correlation-pruned; if interpretability "
            "matters, prefer domain-driven retention over automatic pruning."
        )
    if ctx.clean.duplicates_removed > 0:
        recommendations.append(
            f"{ctx.clean.duplicates_removed} duplicate rows were removed; if duplicates "
            "recur between runs, add upstream uniqueness constraints."
        )
    recommendations.append(
        "Validate downstream model quality with cross-validation on the train split only; "
        "treat val/test as untouched until final reporting."
    )
    recommendations.append(
        "Version the raw inputs and this run's manifest with DVC (or an equivalent "
        "data registry) so every model can be traced to exact bytes."
    )
    return recommendations


def _assumptions(ctx: RunContext) -> list[str]:
    """Documented assumptions, adapted to what the run actually did."""
    assumptions = [
        "Split-first discipline: all fitted statistics were learned on the train "
        "partition only; val/test were transformed, never fitted.",
        "The audit is informational — no fitted decision consumes audit output.",
        "Exact duplicate rows are removed pre-split (stateless, prevents cross-split "
        "straddling); near-duplicates are reported, not removed.",
        "Column-drop decisions (null ceiling, constants, ID-like) were fitted on train "
        "statistics only.",
        "Outlier handling clips values to train-fitted bounds; rows are never dropped "
        "after the split.",
        "Engineered features with non-finite values are replaced with train medians of "
        "the engineered outputs.",
        "Unseen categories in val/test map to all-zero one-hot rows.",
        "The target column was never imputed, scaled, or used in any fit.",
    ]
    if ctx.split_meta.strategy.startswith("chronological"):
        assumptions.append(
            "Lag/rolling features are stateless and past-only (shift ≥ 1); computing "
            "them pre-split gives val/test rows their legitimate history without "
            "look-ahead."
        )
    if ctx.ingest.coercions:
        coerced = [c.column for c in ctx.ingest.coercions if c.unresolved_to_null > 0]
        if coerced:
            assumptions.append(
                f"Non-castable values in {coerced} became null during numeric coercion "
                "and were imputed with train statistics."
            )
    return assumptions


def render_summary(ctx: RunContext) -> str:
    """Render the full ``summary.md`` document."""
    lines: list[str] = []
    lines.append("# Preprocessing Run Summary")
    lines.append("")
    lines.append(
        f"*Run window (UTC):* {ctx.started_at_utc} → {ctx.finished_at_utc} · *Seed:* {ctx.seed}"
    )
    lines.append("")
    lines.append("## 1. Inputs & snapshot")
    lines.append("")
    rows = [[f.path.name, f.rows, f.sha256[:12] + "…"] for f in ctx.ingest.files]
    lines.append(_table(["input file", "rows", "sha256"], rows))
    if ctx.ingest.skipped:
        lines.append(
            _table(
                ["skipped file", "reason"],
                [[f.path.name, f.reason] for f in ctx.ingest.skipped],
            )
        )
    lines.append(
        f"Ingested snapshot: **{ctx.ingest.rows:,} rows × {len(ctx.ingest.columns)} columns**"
        + (f" → `{ctx.ingest.snapshot_path.name}`" if ctx.ingest.snapshot_path else "")
        + "\n"
    )
    coercion_rows = [
        [c.column, c.method, c.target_dtype, c.unresolved_to_null]
        for c in ctx.ingest.coercions
        if c.method not in ("kept_string", "all_null") or c.unresolved_to_null
    ]
    lines.append(_table(["column", "coercion", "dtype", "unresolved→null"], coercion_rows))

    lines.append("## 2. Data-quality findings")
    lines.append("")
    top_null = sorted(
        (c for c in ctx.audit.per_column if c.role != "index"),
        key=lambda c: -c.null_rate,
    )[:10]
    lines.append(
        _table(
            ["column", "null %", "unique", "skew", "flags"],
            [
                [
                    c.name,
                    f"{c.null_rate * 100:.2f}",
                    c.n_unique if c.n_unique is not None else "—",
                    f"{c.skew:.3f}" if c.skew is not None else "—",
                    ", ".join(
                        (["constant"] if c.is_constant else [])
                        + (["near-constant"] if c.is_near_constant else [])
                        + (["id-like"] if c.is_id_like else [])
                        + (["leak?"] if c.leakage_flags else [])
                    )
                    or "—",
                ]
                for c in top_null
                if c.null_rate > 0
                or c.is_constant
                or c.is_near_constant
                or c.is_id_like
                or c.leakage_flags
            ],
        )
    )
    lines.append(f"- Duplicate rows removed: **{ctx.clean.duplicates_removed:,}**")
    lines.append(
        f"- ±inf values nulled: **{sum(ctx.clean.inf_replaced.values()):,}**"
        if ctx.clean.inf_replaced
        else "- ±inf values nulled: **0**"
    )
    if ctx.clean.negatives_nulled:
        lines.append(
            f"- Domain-rule negatives nulled: **{sum(ctx.clean.negatives_nulled.values()):,}**"
        )
    if ctx.clean.null_target_rows_dropped:
        lines.append(f"- Rows with null target dropped: **{ctx.clean.null_target_rows_dropped:,}**")
    if ctx.audit.high_corr_pairs:
        lines.append(
            _table(
                ["high-corr pair", "r"],
                [
                    [f"{p['left']} ~ {p['right']}", f"{p['r']:.4f}"]
                    for p in ctx.audit.high_corr_pairs[:10]
                ],
            )
        )
    if ctx.audit.leakage_flags:
        lines.append(
            _table(
                ["potential leakage column", "reasons"],
                [[f["column"], "; ".join(f["reasons"])] for f in ctx.audit.leakage_flags],
            )
        )
    if ctx.split_meta.near_duplicate_collisions:
        lines.append(
            _table(
                ["near-duplicate collisions (rounded hash)", "count"],
                [[pair, count] for pair, count in ctx.split_meta.near_duplicate_collisions.items()],
            )
        )
    lines.append("")

    lines.append("## 3. Split")
    lines.append("")
    lines.append(
        f"- Strategy: **{ctx.split_meta.strategy}** · sizes "
        f"{ctx.split_meta.sizes} · applied ratios {ctx.split_meta.applied_ratios}"
    )
    if ctx.split_meta.ratios_normalized:
        lines.append(
            f"  - Config ratios {ctx.split_meta.requested_ratios} were renormalized to sum to 1."
        )
    if ctx.split_meta.fallback:
        lines.append(f"  - Fallback: {ctx.split_meta.fallback}")
    if ctx.split_meta.group_overlap_zero is not None:
        lines.append(f"- Zero group overlap verified: **{ctx.split_meta.group_overlap_zero}**")
    if ctx.split_meta.time_ranges:
        for name, rng in ctx.split_meta.time_ranges.items():
            lines.append(f"  - {name}: {rng['min']} → {rng['max']}")
    task = ctx.split_meta.target_task
    if task == "classification":
        for name, dist in ctx.split_meta.target_distribution.items():
            props = ", ".join(f"{k}: {v:.3f}" for k, v in dist.get("class_proportions", {}).items())
            lines.append(f"  - {name} target proportions: {props}")
    lines.append("")

    lines.append("## 4. Fitted decisions (train only)")
    lines.append("")
    pruning = ctx.fitted_stats.get("column_pruning", {})
    dropped = pruning.get("dropped", {})
    lines.append(
        _table(["dropped column", "reason (train-fitted)"], [[k, v] for k, v in dropped.items()])
    )
    imputation = ctx.fitted_stats.get("imputation", {})
    lines.append(
        f"- Imputation: **{imputation.get('strategy', 'n/a')}** on "
        f"{len(imputation.get('fill_values', {}))} numeric column(s); "
        f"{len(ctx.fitted_stats.get('missing_indicators', []))} missing-indicator column(s) added."
    )
    bounds = ctx.fitted_stats.get("outlier_bounds", {})
    lines.append(
        f"- Outliers: **{bounds.get('method', 'n/a')}** bounds fitted for "
        f"{len(bounds.get('bounds', {}))} column(s); clipping counts logged per partition."
    )
    engineering = ctx.fitted_stats.get("feature_engineering", {})
    catalog = engineering.get("catalog", [])
    lines.append(
        _table(
            ["engineered feature", "formula"],
            [[e["name"], e["formula"]] for e in catalog[:25]],
        )
    )
    encoding = ctx.fitted_stats.get("encoding", {})
    lines.append(
        f"- Encoding: one-hot applied to {len(encoding.get('categories', {}))} column(s); "
        f"{len(encoding.get('dropped', {}))} dropped (high cardinality)."
    )
    selection = ctx.fitted_stats.get("selection", {})
    lines.append(f"- Selection: {len(selection.get('dropped', {}))} column(s) dropped.")
    scaling = ctx.fitted_stats.get("scaling", {})
    scale_rows = [
        [name, entry["kind"], entry["reason"]] for name, entry in list(scaling.items())[:25]
    ]
    lines.append(_table(["column", "scaler", "reason"], scale_rows))
    lines.append("")

    lines.append("## 5. Assumptions")
    lines.append("")
    for assumption in _assumptions(ctx):
        lines.append(f"- {assumption}")
    lines.append("")

    lines.append("## 6. Recommended next steps")
    lines.append("")
    for recommendation in _recommendations(ctx):
        lines.append(f"- {recommendation}")
    lines.append("")

    lines.append("## 7. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# deterministic rerun (same input + config + image ⇒ byte-identical outputs)")
    lines.append("make run   # or: numprep run --config <cfg> --input <dir> --output <dir>")
    lines.append("```")
    lines.append("")
    output_rows = [[name, info] for name, info in ctx.output_files.items()]
    lines.append(_table(["output", "sha256"], output_rows))
    if ctx.manifest_sha256:
        lines.append(f"Manifest SHA-256: `{ctx.manifest_sha256}`")
    lines.append("")
    versions = ", ".join(f"{k} {v}" for k, v in library_versions().items())
    lines.append(f"Library context: {versions}")
    lines.append("")
    return "\n".join(lines)
