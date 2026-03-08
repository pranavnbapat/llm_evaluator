#!/usr/bin/env python3
"""
Generate context-evaluation insights report from DB/XLSX artifacts.

Single-run mode:
  python insights/generate_context_insights.py --run-dir <run_dir>

Bulk mode (default):
  python insights/generate_context_insights.py
This scans results/runs/*/* and generates only missing report artifacts.
"""

from __future__ import annotations

import sqlite3
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

SCORE_COLS = [
    "relevance",
    "factual_accuracy",
    "completeness",
    "fluency",
    "coherence",
    "prompt_alignment",
    "token_efficiency",
    "overall_quality",
]


@dataclass
class Inputs:
    scores: pd.DataFrame
    results: pd.DataFrame
    scores_xlsx_rows: int
    results_xlsx_rows: int
    results_by_model_all_rows: int


@dataclass
class RunPaths:
    run_dir: Path
    scores_db: Path
    results_db: Path
    scores_xlsx: Path
    results_xlsx: Path
    results_by_model_xlsx: Path
    out_md: Path


def _base_question_id(qid: str) -> str:
    if isinstance(qid, str) and "_" in qid:
        return qid.split("_", 1)[0]
    return str(qid)


def _fmt(x: float, n: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{n}f}"


def _load_sqlite_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)


def load_inputs(paths: RunPaths) -> Inputs:
    required = [
        paths.scores_db,
        paths.results_db,
        paths.scores_xlsx,
        paths.results_xlsx,
        paths.results_by_model_xlsx,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n- " + "\n- ".join(missing))

    scores = _load_sqlite_table(paths.scores_db, "scores")
    results = _load_sqlite_table(paths.results_db, "evaluations")

    scores_xlsx_rows = len(pd.read_excel(paths.scores_xlsx, sheet_name=0))
    results_xlsx_rows = len(pd.read_excel(paths.results_xlsx, sheet_name=0))

    by_model_xl = pd.ExcelFile(paths.results_by_model_xlsx)
    if "all_results" in by_model_xl.sheet_names:
        results_by_model_all_rows = len(pd.read_excel(paths.results_by_model_xlsx, sheet_name="all_results"))
    else:
        results_by_model_all_rows = -1

    return Inputs(
        scores=scores,
        results=results,
        scores_xlsx_rows=scores_xlsx_rows,
        results_xlsx_rows=results_xlsx_rows,
        results_by_model_all_rows=results_by_model_all_rows,
    )


def _to_md_table(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_No data_"
    if index:
        df = df.reset_index()
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def build_report(inp: Inputs) -> str:
    scores = inp.scores.copy()
    results = inp.results.copy()

    scores["base_qid"] = scores["question_id"].map(_base_question_id)
    results["base_qid"] = results["question_id"].map(_base_question_id)

    total_scores = len(scores)
    total_results = len(results)
    model_count = scores["model_name"].nunique()
    language_count = scores["language"].nunique()
    question_count = scores["base_qid"].nunique()
    runs_per_cell = scores.groupby(["model_name", "language", "base_qid"]).size().value_counts().to_dict()

    # Data quality checks
    out_of_range_counts: Dict[str, int] = {}
    for c in SCORE_COLS:
        out_of_range_counts[c] = int(((scores[c] < 0) | (scores[c] > 1)).sum())
    null_counts = scores[SCORE_COLS].isna().sum().to_dict()

    # Model ranking
    model_summary = (
        scores.groupby("model_name")
        .agg(
            n=("id", "count"),
            avg_overall=("overall_quality", "mean"),
            std_overall=("overall_quality", "std"),
            p10=("overall_quality", lambda x: np.percentile(x, 10)),
            p90=("overall_quality", lambda x: np.percentile(x, 90)),
        )
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )
    for c in ["avg_overall", "std_overall", "p10", "p90"]:
        model_summary[c] = model_summary[c].map(lambda x: float(x))

    # Per-metric by model
    metric_by_model = (
        scores.groupby("model_name")[SCORE_COLS]
        .mean()
        .sort_values("overall_quality", ascending=False)
        .reset_index()
    )

    # Language analysis
    lang_summary = (
        scores.groupby("language")
        .agg(
            n=("id", "count"),
            avg_overall=("overall_quality", "mean"),
            std_overall=("overall_quality", "std"),
        )
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )
    top_lang = lang_summary.head(5).copy()
    bottom_lang = lang_summary.tail(5).sort_values("avg_overall", ascending=True).copy()

    # Question analysis
    q_summary = (
        scores.groupby("base_qid")
        .agg(
            n=("id", "count"),
            avg_overall=("overall_quality", "mean"),
            avg_factual=("factual_accuracy", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_fluency=("fluency", "mean"),
        )
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )

    # Latency analysis from results DB
    latency_model = (
        results.groupby("model_name")
        .agg(
            n=("id", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            p90_latency_ms=("latency_ms", lambda x: np.percentile(x, 90)),
        )
        .sort_values("avg_latency_ms", ascending=True)
        .reset_index()
    )

    # Run stability: std across 3 runs for each model-language-question triplet
    cell_std = (
        scores.groupby(["model_name", "language", "base_qid"])["overall_quality"]
        .std()
        .reset_index(name="run_std")
    )
    stability = (
        cell_std.groupby("model_name")["run_std"]
        .agg(["mean", "median", "max"])
        .sort_values("mean", ascending=True)
        .reset_index()
    )

    # Completion checks
    results_with_resp = int(results["response"].notna().sum())
    coverage_scores_vs_results = (total_scores / results_with_resp) if results_with_resp else 0.0

    # Extremes
    best_rows = scores.nlargest(10, "overall_quality")[
        ["model_name", "language", "question_id", "overall_quality", "factual_accuracy", "fluency"]
    ]
    worst_rows = scores.nsmallest(10, "overall_quality")[
        ["model_name", "language", "question_id", "overall_quality", "factual_accuracy", "fluency"]
    ]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Context Evaluation Insights Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Total scored responses: **{total_scores}**")
    lines.append(f"- Total evaluation responses (source DB): **{results_with_resp}**")
    lines.append(f"- Coverage (scores / results): **{coverage_scores_vs_results:.2%}**")
    lines.append(f"- Models evaluated: **{model_count}**")
    lines.append(f"- Languages covered: **{language_count}**")
    lines.append(f"- Question families: **{question_count}**")
    if not model_summary.empty:
        top_model = model_summary.iloc[0]
        lines.append(
            f"- Best model by average overall quality: **{top_model['model_name']}** "
            f"(**{_fmt(top_model['avg_overall'])}**)"
        )
    lines.append("")
    lines.append("## Data Integrity Checks")
    lines.append("")
    lines.append(f"- `evaluation_scores_euf_context.db` rows: **{total_scores}**")
    lines.append(f"- `evaluation_scores_euf_context.xlsx` rows: **{inp.scores_xlsx_rows}**")
    lines.append(f"- `evaluation_results_euf_context.db` rows: **{total_results}**")
    lines.append(f"- `evaluation_results_euf_context.xlsx` rows: **{inp.results_xlsx_rows}**")
    if inp.results_by_model_all_rows >= 0:
        lines.append(f"- `evaluation_results_euf_context_by_model.xlsx` (`all_results`) rows: **{inp.results_by_model_all_rows}**")
    lines.append(f"- Run-count distribution per (model, language, question): **{runs_per_cell}**")
    lines.append("")
    lines.append("### Score Range Validation")
    lines.append("")
    range_df = pd.DataFrame(
        {
            "metric": SCORE_COLS,
            "out_of_range_count": [out_of_range_counts[c] for c in SCORE_COLS],
            "null_count": [int(null_counts[c]) for c in SCORE_COLS],
        }
    )
    lines.append(_to_md_table(range_df))
    lines.append("")

    lines.append("## Model Ranking")
    lines.append("")
    rank_df = model_summary.copy()
    for c in ["avg_overall", "std_overall", "p10", "p90"]:
        rank_df[c] = rank_df[c].map(lambda x: _fmt(x))
    lines.append(_to_md_table(rank_df))
    lines.append("")

    lines.append("## Metric Breakdown by Model")
    lines.append("")
    mbm = metric_by_model.copy()
    for c in SCORE_COLS:
        mbm[c] = mbm[c].map(lambda x: _fmt(x))
    lines.append(_to_md_table(mbm))
    lines.append("")

    lines.append("## Language Insights")
    lines.append("")
    top_lang_fmt = top_lang.copy()
    bottom_lang_fmt = bottom_lang.copy()
    for df in [top_lang_fmt, bottom_lang_fmt]:
        for c in ["avg_overall", "std_overall"]:
            df[c] = df[c].map(lambda x: _fmt(x))
    lines.append("### Top 5 Languages (avg overall quality)")
    lines.append("")
    lines.append(_to_md_table(top_lang_fmt))
    lines.append("")
    lines.append("### Bottom 5 Languages (avg overall quality)")
    lines.append("")
    lines.append(_to_md_table(bottom_lang_fmt))
    lines.append("")

    lines.append("## Question-Level Insights")
    lines.append("")
    qf = q_summary.copy()
    for c in ["avg_overall", "avg_factual", "avg_completeness", "avg_fluency"]:
        qf[c] = qf[c].map(lambda x: _fmt(x))
    lines.append(_to_md_table(qf))
    lines.append("")

    lines.append("## Latency by Model (from Results DB)")
    lines.append("")
    ldf = latency_model.copy()
    for c in ["avg_latency_ms", "p90_latency_ms"]:
        ldf[c] = ldf[c].map(lambda x: _fmt(x, 1))
    lines.append(_to_md_table(ldf))
    lines.append("")

    lines.append("## Run-to-Run Stability (within model-language-question)")
    lines.append("")
    sdf = stability.copy()
    for c in ["mean", "median", "max"]:
        sdf[c] = sdf[c].map(lambda x: _fmt(x, 4))
    lines.append(_to_md_table(sdf))
    lines.append("")

    lines.append("## Best and Worst Scored Responses (Diagnostic)")
    lines.append("")
    br = best_rows.copy()
    wr = worst_rows.copy()
    for df in [br, wr]:
        for c in ["overall_quality", "factual_accuracy", "fluency"]:
            df[c] = df[c].map(lambda x: _fmt(x))
    lines.append("### Top 10")
    lines.append("")
    lines.append(_to_md_table(br))
    lines.append("")
    lines.append("### Bottom 10")
    lines.append("")
    lines.append(_to_md_table(wr))
    lines.append("")

    lines.append("## Key Insights")
    lines.append("")
    if not model_summary.empty:
        best = model_summary.iloc[0]
        worst = model_summary.iloc[-1]
        delta = float(best["avg_overall"] - worst["avg_overall"])
        lines.append(
            f"- Performance spread across models is **{_fmt(delta)}** "
            f"(best `{best['model_name']}` vs worst `{worst['model_name']}`)."
        )
    if not lang_summary.empty:
        ldelta = float(lang_summary["avg_overall"].max() - lang_summary["avg_overall"].min())
        lines.append(f"- Language spread is **{_fmt(ldelta)}** between highest and lowest average language scores.")
    lines.append("- Completeness is strongly affected by context-coverage behavior and may dominate question-level differences.")
    lines.append("- Factual-accuracy values are generally high due to NLI/context matching; inspect low outliers for grounding failures.")
    lines.append("- This report is deterministic from artifact files and can be regenerated after each scoring run.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        run_dir=run_dir,
        scores_db=run_dir / "scores" / "evaluation_scores_euf_context.db",
        results_db=run_dir / "raw" / "evaluation_results_euf_context.db",
        scores_xlsx=run_dir / "scores" / "evaluation_scores_euf_context.xlsx",
        results_xlsx=run_dir / "raw" / "evaluation_results_euf_context.xlsx",
        results_by_model_xlsx=run_dir / "raw" / "evaluation_results_euf_context_by_model.xlsx",
        out_md=run_dir / "insights" / "EVALUATION_CONTEXT_REPORT.md",
    )


def _discover_run_dirs(repo_root: Path) -> Iterable[Path]:
    runs_root = repo_root / "results" / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.glob("*/*") if p.is_dir()])


def _is_complete(paths: RunPaths) -> bool:
    return paths.out_md.exists()


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    paths = _run_paths(run_dir)
    if not force and _is_complete(paths):
        return False, f"skip (already complete): {run_dir}"
    try:
        inp = load_inputs(paths)
    except FileNotFoundError as e:
        return False, f"skip (missing input): {run_dir} :: {e}"

    try:
        paths.out_md.parent.mkdir(parents=True, exist_ok=True)
        report = build_report(inp)
        paths.out_md.write_text(report, encoding="utf-8")
    except Exception as e:
        return False, f"skip (generation error): {run_dir} :: {e}"
    return True, f"generated: {run_dir}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate context insights report for one run or all runs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific run directory (e.g., results/runs/a40/<run_id>).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all runs under results/runs/*/* (default when --run-dir is omitted).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when output artifact already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        changed, msg = _process_run(run_dir, force=args.force)
        print(msg)
        if changed:
            print(f"Wrote: {_run_paths(run_dir).out_md}")
        return

    run_dirs = list(_discover_run_dirs(ROOT))
    if not run_dirs:
        print("No run directories found under results/runs")
        return

    generated = 0
    skipped = 0
    for run_dir in run_dirs:
        changed, msg = _process_run(run_dir, force=args.force)
        print(msg)
        if changed:
            generated += 1
        else:
            skipped += 1

    print(f"Summary: generated={generated}, skipped={skipped}, total={len(run_dirs)}")


if __name__ == "__main__":
    main()
