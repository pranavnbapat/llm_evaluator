#!/usr/bin/env python3
"""
Generate one GPU-level context-evaluation report by aggregating all runs under:
  results/runs/<gpu>/*

Example:
  python insights/generate_gpu_insights_report.py --gpu a40
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translations.eu_24_languages_euf_context import get_all_questions_with_context

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

LANGUAGE_NAMES = {
    "BG": "Bulgarian",
    "HR": "Croatian",
    "CS": "Czech",
    "DA": "Danish",
    "NL": "Dutch",
    "EN": "English",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "DE": "German",
    "EL": "Greek",
    "HU": "Hungarian",
    "GA": "Irish",
    "IT": "Italian",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "MT": "Maltese",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "SK": "Slovak",
    "SL": "Slovenian",
    "ES": "Spanish",
    "SV": "Swedish",
}


def _iter_run_dirs(gpu_runs_dir: Path) -> list[Path]:
    return sorted([p for p in gpu_runs_dir.iterdir() if p.is_dir()])


def _read_scores(scores_db: Path) -> pd.DataFrame:
    if not scores_db.exists():
        return pd.DataFrame(columns=["model_name", "overall_quality", "language", "question_id"])
    with sqlite3.connect(scores_db) as con:
        return pd.read_sql_query(
            "SELECT model_name, overall_quality, language, question_id FROM scores",
            con,
        )


def _read_results(results_db: Path) -> pd.DataFrame:
    if not results_db.exists():
        return pd.DataFrame(columns=["model_name", "latency_ms", "response"])
    with sqlite3.connect(results_db) as con:
        return pd.read_sql_query(
            "SELECT model_name, latency_ms, response FROM evaluations",
            con,
        )


def _fmt_model(model_name: str | None, value: float | None, suffix: str = "") -> str:
    if not model_name:
        return "N/A"
    if value is None:
        return model_name
    if suffix:
        return f"{model_name} ({value:.3f} {suffix})"
    return f"{model_name} ({value:.3f})"


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _to_md_table_fmt(df: pd.DataFrame, float_cols: list[str], ndigits: int = 3) -> str:
    if df.empty:
        return "_No data_"
    d = df.copy()
    for c in float_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda x: f"{float(x):.{ndigits}f}" if pd.notna(x) else "NA")
    return _to_md_table(d)


def _read_xlsx_rows(path: Path, sheet_name: str | int = 0) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_excel(path, sheet_name=sheet_name)))
    except Exception:
        return 0


def _parse_generation_profile(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    lines = config_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "generation_profile:" and not line.startswith(" "):
            start = i
            break
    if start is None:
        return {}

    out: dict = {}
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        if not line.startswith("  "):
            break
        m = line.strip().split(":", 1)
        if len(m) != 2:
            continue
        key = m[0].strip()
        val = m[1].strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if val.isdigit():
            out[key] = int(val)
        else:
            out[key] = val
    return out


def build_report(gpu: str, run_dirs: list[Path], generation_profile: dict) -> str:
    all_scores: list[pd.DataFrame] = []
    all_results: list[pd.DataFrame] = []
    scores_xlsx_rows = 0
    results_xlsx_rows = 0
    results_by_model_all_rows = 0

    for run_dir in run_dirs:
        scores_db = run_dir / "scores" / "evaluation_scores_euf_context.db"
        results_db = run_dir / "raw" / "evaluation_results_euf_context.db"
        all_scores.append(_read_scores(scores_db))
        all_results.append(_read_results(results_db))
        scores_xlsx_rows += _read_xlsx_rows(run_dir / "scores" / "evaluation_scores_euf_context.xlsx", 0)
        results_xlsx_rows += _read_xlsx_rows(run_dir / "raw" / "evaluation_results_euf_context.xlsx", 0)
        results_by_model_all_rows += _read_xlsx_rows(
            run_dir / "raw" / "evaluation_results_euf_context_by_model.xlsx",
            "all_results",
        )

    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    total_scored = int(len(scores))
    total_results = int(len(results))
    total_eval_responses = int(results["response"].notna().sum()) if not results.empty and "response" in results.columns else 0
    coverage = (total_scored / total_eval_responses) if total_eval_responses else 0.0

    if not scores.empty and "model_name" in scores.columns:
        model_names = sorted([m for m in scores["model_name"].dropna().unique().tolist()])
    elif not results.empty and "model_name" in results.columns:
        model_names = sorted([m for m in results["model_name"].dropna().unique().tolist()])
    else:
        model_names = []
    model_count = len(model_names)

    all_questions = get_all_questions_with_context()
    language_count = len({q["language"] for q in all_questions})
    question_family_count = len({str(q["question_id"]).split("_", 1)[0] for q in all_questions})
    model_labels = {m: f"M{i}" for i, m in enumerate(model_names, start=1)}

    runs_per_cell: dict[int, int] = {}
    score_range_df = pd.DataFrame(columns=["metric", "out_of_range_count", "null_count"])
    if not scores.empty:
        scores["base_qid"] = scores["question_id"].map(
            lambda q: str(q).split("_", 1)[0] if isinstance(q, str) and "_" in q else str(q)
        )
        if {"model_name", "language", "base_qid"}.issubset(scores.columns):
            runs_per_cell = (
                scores.groupby(["model_name", "language", "base_qid"])
                .size()
                .value_counts()
                .sort_index()
                .to_dict()
            )
        out_of_range = []
        nulls = []
        for c in SCORE_COLS:
            if c in scores.columns:
                out_of_range.append(int(((scores[c] < 0) | (scores[c] > 1)).sum()))
                nulls.append(int(scores[c].isna().sum()))
            else:
                out_of_range.append(0)
                nulls.append(0)
        score_range_df = pd.DataFrame(
            {
                "metric": SCORE_COLS,
                "out_of_range_count": out_of_range,
                "null_count": nulls,
            }
        )

    best_model_name: str | None = None
    best_model_score: float | None = None
    fastest_model_name: str | None = None
    fastest_model_latency: float | None = None
    fastest_good_model_name: str | None = None
    fastest_good_latency: float | None = None
    model_ranking = pd.DataFrame(columns=["model_name", "n", "avg_overall", "std_overall", "p10", "p90"])
    metric_by_model = pd.DataFrame(
        columns=[
            "model_name",
            "relevance",
            "factual_accuracy",
            "completeness",
            "fluency",
            "coherence",
            "prompt_alignment",
            "token_efficiency",
            "overall_quality",
        ]
    )
    lang_summary = pd.DataFrame(columns=["language", "n", "avg_overall", "std_overall"])
    q_summary = pd.DataFrame(
        columns=[
            "base_qid",
            "n",
            "avg_overall",
            "avg_factual",
            "avg_completeness",
            "avg_fluency",
        ]
    )
    latency_model = pd.DataFrame(columns=["model_name", "n", "avg_latency_ms", "p90_latency_ms"])
    best_rows = pd.DataFrame(columns=["model_name", "language", "question_id", "overall_quality", "factual_accuracy", "fluency"])
    worst_rows = pd.DataFrame(columns=["model_name", "language", "question_id", "overall_quality", "factual_accuracy", "fluency"])

    if not scores.empty:
        model_ranking = (
            scores.groupby("model_name")
            .agg(
                n=("overall_quality", "count"),
                avg_overall=("overall_quality", "mean"),
                std_overall=("overall_quality", "std"),
                p10=("overall_quality", lambda x: np.percentile(x, 10)),
                p90=("overall_quality", lambda x: np.percentile(x, 90)),
            )
            .sort_values("avg_overall", ascending=False)
            .reset_index()
        )
        present_score_cols = [c for c in SCORE_COLS if c in scores.columns]
        if present_score_cols:
            metric_by_model = (
                scores.groupby("model_name")[present_score_cols]
                .mean()
                .sort_values("overall_quality", ascending=False)
                .reset_index()
            )
        needed_diag_cols = ["model_name", "language", "question_id", "overall_quality", "factual_accuracy", "fluency"]
        if all(c in scores.columns for c in needed_diag_cols):
            best_rows = scores.nlargest(10, "overall_quality")[needed_diag_cols].copy()
            worst_rows = scores.nsmallest(10, "overall_quality")[needed_diag_cols].copy()
        if {"language", "overall_quality"}.issubset(scores.columns):
            lang_summary = (
                scores.groupby("language")
                .agg(
                    n=("overall_quality", "count"),
                    avg_overall=("overall_quality", "mean"),
                    std_overall=("overall_quality", "std"),
                )
                .sort_values("avg_overall", ascending=False)
                .reset_index()
            )
        if {"base_qid", "overall_quality", "factual_accuracy", "completeness", "fluency"}.issubset(scores.columns):
            q_summary = (
                scores.groupby("base_qid")
                .agg(
                    n=("overall_quality", "count"),
                    avg_overall=("overall_quality", "mean"),
                    avg_factual=("factual_accuracy", "mean"),
                    avg_completeness=("completeness", "mean"),
                    avg_fluency=("fluency", "mean"),
                )
                .sort_values("avg_overall", ascending=False)
                .reset_index()
            )

        model_quality = (
            scores.groupby("model_name", dropna=True)["overall_quality"]
            .mean()
            .sort_values(ascending=False)
        )
        if not model_quality.empty:
            best_model_name = str(model_quality.index[0])
            best_model_score = float(model_quality.iloc[0])

        if not results.empty and "latency_ms" in results.columns:
            lat = results[results["latency_ms"].notna()].copy()
            if not lat.empty:
                model_latency = lat.groupby("model_name", dropna=True)["latency_ms"].mean()
                quality_threshold = float(model_quality.median()) if len(model_quality) > 0 else None
                if quality_threshold is not None:
                    good_models = set(model_quality[model_quality >= quality_threshold].index.tolist())
                    candidate_latency = model_latency[model_latency.index.isin(good_models)]
                else:
                    candidate_latency = model_latency
                if candidate_latency.empty:
                    candidate_latency = model_latency
                if not candidate_latency.empty:
                    fastest_good_model_name = str(candidate_latency.sort_values().index[0])
                    fastest_good_latency = float(candidate_latency.sort_values().iloc[0])

    if not results.empty and {"model_name", "latency_ms"}.issubset(results.columns):
        lat = results[results["latency_ms"].notna()].copy()
        if not lat.empty:
            latency_model = (
                lat.groupby("model_name")
                .agg(
                    n=("latency_ms", "count"),
                    avg_latency_ms=("latency_ms", "mean"),
                    p90_latency_ms=("latency_ms", lambda x: np.percentile(x, 90)),
                )
                .sort_values("avg_latency_ms", ascending=True)
                .reset_index()
            )
            fastest_model_name = str(latency_model.iloc[0]["model_name"])
            fastest_model_latency = float(latency_model.iloc[0]["avg_latency_ms"])

    lines: list[str] = []
    lines.append("# Context Evaluation Insights Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Total scored responses: **{total_scored}**")
    lines.append(f"- Total evaluation responses (source DB): **{total_eval_responses}**")
    lines.append(f"- Coverage (scores / results): **{coverage:.2%}**")
    lines.append(f"- Models evaluated: **{model_count}**")
    lines.append(f"- Languages covered: **{language_count}**")
    lines.append(f"- Question families: **{question_family_count}**")
    lines.append(f"- Best model by average overall quality: **{_fmt_model(best_model_name, best_model_score)}**")
    lines.append(
        "- Fastest responding model (lowest average latency): "
        f"**{_fmt_model(fastest_model_name, fastest_model_latency, 'ms avg latency')}**"
    )
    lines.append(
        "- Best latency among quality-qualified models: "
        f"**{_fmt_model(fastest_good_model_name, fastest_good_latency, 'ms avg latency')}**"
    )
    lines.append("")
    lines.append("## Run Configuration Assumptions")
    lines.append("")
    lines.append(
        f"- Config source: **gpu_runtime/config.yaml**"
    )
    lines.append(
        f"- Assumed concurrent users: **{generation_profile.get('concurrent_users', 'N/A')}**"
    )
    lines.append(
        f"- Target max output tokens: **{generation_profile.get('target_max_output_tokens', 'N/A')}**"
    )
    lines.append(
        f"- GPU profile in config: **{generation_profile.get('gpu', gpu)}**"
    )
    if generation_profile.get("generated_at"):
        lines.append(f"- Profile generated at: **{generation_profile.get('generated_at')}**")
    lines.append(
        "- Context window policy: **max_model_len is concurrency-sized per model; "
        "usable_input_tokens = max_model_len - target_max_output_tokens**"
    )
    lines.append("")
    lines.append("### Top Models by Criterion")
    lines.append("")
    lines.append(
        f"- Highest overall quality: **{_fmt_model(best_model_name, best_model_score)}**"
    )
    lines.append(
        f"- Fastest response (lowest avg latency): **{_fmt_model(fastest_model_name, fastest_model_latency, 'ms avg latency')}**"
    )
    lines.append(
        f"- Fastest model among quality-qualified set: **{_fmt_model(fastest_good_model_name, fastest_good_latency, 'ms avg latency')}**"
    )
    lines.append(
        "_Quality-qualified set uses models with average overall quality >= median model quality in this aggregate._"
    )
    lines.append("")
    lines.append("### Token Budget Notes")
    lines.append("")
    lines.append(
        "- `seq_lens` are candidate total context-window sizes tested during config generation "
        "(e.g., 4096, 8192, 16384)."
    )
    lines.append(
        "- Multiple values are used as a search grid: for each model, the generator selects the "
        "largest candidate that still fits GPU and concurrency constraints."
    )
    lines.append(
        "- `target_max_output_tokens` reserves generation budget inside `max_model_len`; "
        "higher output caps reduce safe input budget and can reduce concurrency."
    )
    lines.append(
        "- Practical interpretation for `512` output tokens: roughly **350-420 English words**, "
        "about **0.7-1.0 A4 page** of plain single-spaced text."
    )
    lines.append("")
    lines.append("## Data Integrity Checks")
    lines.append("")
    lines.append(f"- `evaluation_scores_euf_context.db` rows: **{total_scored}**")
    lines.append(f"- `evaluation_scores_euf_context.xlsx` rows: **{scores_xlsx_rows}**")
    lines.append(f"- `evaluation_results_euf_context.db` rows: **{total_results}**")
    lines.append(f"- `evaluation_results_euf_context.xlsx` rows: **{results_xlsx_rows}**")
    lines.append(
        "- `evaluation_results_euf_context_by_model.xlsx` (`all_results`) rows: "
        f"**{results_by_model_all_rows}**"
    )
    lines.append(f"- Run-count distribution per (model, language, question): **{runs_per_cell}**")
    lines.append("")
    lines.append("### Score Range Validation")
    lines.append("")
    lines.append(_to_md_table(score_range_df))
    lines.append("")
    lines.append("## Model Ranking")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            model_ranking,
            float_cols=["avg_overall", "std_overall", "p10", "p90"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("## Metric Breakdown by Model")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            metric_by_model,
            float_cols=SCORE_COLS,
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("### Model Labels Used in Tables")
    lines.append("")
    label_df = pd.DataFrame(
        [{"Label": model_labels[m], "Model Name": m} for m in model_names]
    )
    lines.append(_to_md_table(label_df))
    lines.append("")
    lines.append("### 3.3 Language Performance (All 24 EU Languages, All Evaluated Models)")
    lines.append("")
    if not scores.empty and {"language", "model_name", "overall_quality"}.issubset(scores.columns) and model_names:
        lang_model = (
            scores.groupby(["language", "model_name"])["overall_quality"]
            .mean()
            .unstack("model_name")
        )
        for m in model_names:
            if m not in lang_model.columns:
                lang_model[m] = pd.NA
        lang_model = lang_model[model_names]
        lang_model["Avg"] = lang_model.mean(axis=1, numeric_only=True)
        lang_model = lang_model.sort_values("Avg", ascending=False)
        rows = []
        for rank, (code, row) in enumerate(lang_model.iterrows(), start=1):
            item = {
                "Rank": rank,
                "Language": LANGUAGE_NAMES.get(code, code),
                "Code": code,
            }
            for m in model_names:
                item[model_labels[m]] = row[m]
            item["Avg"] = row["Avg"]
            rows.append(item)
        lang_perf_df = pd.DataFrame(rows)
        float_cols = [model_labels[m] for m in model_names] + ["Avg"]
        lines.append(_to_md_table_fmt(lang_perf_df, float_cols=float_cols, ndigits=3))
    else:
        lines.append("_No data_")
    lines.append("")
    lines.append("### 3.4 Detailed Metric Breakdown (All Evaluated Models)")
    lines.append("")
    if not metric_by_model.empty and model_labels:
        mb = metric_by_model.copy()
        mb["Model"] = mb["model_name"].map(model_labels)
        ordered_cols = [
            "Model",
            "relevance",
            "factual_accuracy",
            "completeness",
            "fluency",
            "coherence",
            "prompt_alignment",
            "token_efficiency",
            "overall_quality",
        ]
        for c in ordered_cols:
            if c not in mb.columns:
                mb[c] = pd.NA
        mb = mb[ordered_cols]
        mb = mb.sort_values("Model", key=lambda s: s.str.extract(r"(\d+)").astype(int)[0])
        mb = mb.rename(
            columns={
                "factual_accuracy": "Factual",
                "completeness": "Complete",
                "fluency": "Fluency",
                "coherence": "Coherence",
                "prompt_alignment": "Alignment",
                "token_efficiency": "Efficiency",
                "overall_quality": "Overall",
                "relevance": "Relevance",
            }
        )
        lines.append(
            _to_md_table_fmt(
                mb,
                float_cols=[
                    "Relevance",
                    "Factual",
                    "Complete",
                    "Fluency",
                    "Coherence",
                    "Alignment",
                    "Efficiency",
                    "Overall",
                ],
                ndigits=3,
            )
        )
    else:
        lines.append("_No data_")
    lines.append("")
    lines.append("## Language Insights")
    lines.append("")
    lines.append("### Top 10 Languages (avg overall quality)")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            lang_summary.head(10),
            float_cols=["avg_overall", "std_overall"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("### Bottom 10 Languages (avg overall quality)")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            lang_summary.tail(10).sort_values("avg_overall", ascending=True),
            float_cols=["avg_overall", "std_overall"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("### Language Insights Interpretation")
    lines.append("")
    if not lang_summary.empty:
        top_lang = lang_summary.iloc[0]
        bottom_lang = lang_summary.iloc[-1]
        spread = float(top_lang["avg_overall"] - bottom_lang["avg_overall"])
        median_std = float(lang_summary["std_overall"].median()) if lang_summary["std_overall"].notna().any() else 0.0
        high_var = lang_summary.sort_values("std_overall", ascending=False).head(3)["language"].tolist()
        lines.append(
            f"- Quality spread across languages is **{spread:.3f}** "
            f"(top: `{top_lang['language']}` {float(top_lang['avg_overall']):.3f}, "
            f"bottom: `{bottom_lang['language']}` {float(bottom_lang['avg_overall']):.3f})."
        )
        lines.append(
            f"- Typical variability by language (median std) is **{median_std:.3f}**; "
            "higher std suggests less consistent behavior across runs/questions."
        )
        if high_var:
            lines.append(
                f"- Highest-variance languages in this aggregate: **{', '.join(high_var)}**; "
                "these are good candidates for targeted prompt/context tuning."
            )
        lines.append(
            "- Operationally, prioritize regression checks for bottom-ranked and high-variance languages "
            "before promoting a model to production."
        )
    else:
        lines.append("- Language-level score data is not available in aggregated score DBs.")
    lines.append("")
    lines.append("## Question-Level Insights")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            q_summary,
            float_cols=["avg_overall", "avg_factual", "avg_completeness", "avg_fluency"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("## Latency by Model (from Results DB)")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            latency_model,
            float_cols=["avg_latency_ms", "p90_latency_ms"],
            ndigits=1,
        )
    )
    lines.append("")
    lines.append("## Best and Worst Scored Responses (Diagnostic)")
    lines.append("")
    lines.append("### Top 10")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            best_rows,
            float_cols=["overall_quality", "factual_accuracy", "fluency"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("### Bottom 10")
    lines.append("")
    lines.append(
        _to_md_table_fmt(
            worst_rows,
            float_cols=["overall_quality", "factual_accuracy", "fluency"],
            ndigits=3,
        )
    )
    lines.append("")
    lines.append("## Key Insights")
    lines.append("")
    if not model_ranking.empty:
        best_model = model_ranking.iloc[0]
        worst_model = model_ranking.iloc[-1]
        spread = float(best_model["avg_overall"] - worst_model["avg_overall"])
        lines.append(
            f"- Performance spread across models is **{spread:.3f}** "
            f"(best `{best_model['model_name']}` vs worst `{worst_model['model_name']}`)."
        )
    if not lang_summary.empty:
        lspread = float(lang_summary["avg_overall"].max() - lang_summary["avg_overall"].min())
        lines.append(f"- Language spread is **{lspread:.3f}** between highest and lowest average language scores.")
    lines.append("- Completeness is strongly affected by context-coverage behavior and may dominate question-level differences.")
    lines.append("- Factual-accuracy values are generally high due to NLI/context matching; inspect low outliers for grounding failures.")
    lines.append("- This report is deterministic from artifact files and can be regenerated after each scoring run.")
    lines.append("")
    lines.append("## Aggregation Scope")
    lines.append("")
    lines.append(f"- GPU bucket: **{gpu}**")
    lines.append(
        "- Concurrency sizing profile: "
        f"**{generation_profile.get('gpu', gpu)} / {generation_profile.get('concurrent_users', 'N/A')} users / "
        f"{generation_profile.get('target_max_output_tokens', 'N/A')} output tokens**"
    )
    lines.append(f"- Run folders discovered: **{len(run_dirs)}**")
    for d in run_dirs:
        lines.append(f"- `{d.name}`")
    lines.append("")
    lines.append(
        "_Note: 'fastest ... with good or best speed' is selected as lowest average latency among models "
        "with quality >= median model quality; if none qualify, lowest latency overall is used._"
    )
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one aggregate context report for a GPU bucket.")
    p.add_argument("--gpu", required=True, help="GPU bucket folder name under results/runs (e.g., a40, a100, b200).")
    p.add_argument(
        "--config-file",
        type=Path,
        default=Path("gpu_runtime/config.yaml"),
        help="Config file containing generation_profile metadata.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report path. Default: results/runs/<gpu>/CONTEXT_EVALUATION_INSIGHTS_REPORT.md",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gpu = str(args.gpu).strip()
    gpu_runs_dir = ROOT / "results" / "runs" / gpu

    if not gpu_runs_dir.exists():
        raise SystemExit(f"GPU runs directory not found: {gpu_runs_dir}")

    run_dirs = _iter_run_dirs(gpu_runs_dir)
    generation_profile = _parse_generation_profile(args.config_file)
    out_path = args.output or (gpu_runs_dir / "CONTEXT_EVALUATION_INSIGHTS_REPORT.md")
    report = build_report(gpu=gpu, run_dirs=run_dirs, generation_profile=generation_profile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
