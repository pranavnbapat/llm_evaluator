#!/usr/bin/env python3
"""
Generate one GPU-level context-evaluation report by aggregating all runs under:
  results/runs/<gpu>/*

Example:
  python insights/generate_gpu_insights_report.py --gpu a40
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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

GPU_REQUIRED_COLS = {"timestamp", "phase", "util_gpu_pct", "mem_used_mb", "temp_c"}

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


def _load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _resolve_chat_completions_url(base: str) -> str:
    u = base.rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    if u.endswith("/v1"):
        return f"{u}/chat/completions"
    return f"{u}/v1/chat/completions"


class _LLMClient:
    def __init__(self, url: str, model: str, api_key: str, timeout_s: int = 90):
        self.url = _resolve_chat_completions_url(url)
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    @classmethod
    def from_repo_env(cls, repo_root: Path):
        env = _load_env(repo_root / ".env")
        llm_url = env.get("LLM_URL", "")
        llm_model = env.get("LLM", "")
        api_key = env.get("LLM_API_KEY", "") or env.get("OPENAI_API_KEY", "")
        if not llm_url or not llm_model:
            return None
        return cls(llm_url, llm_model, api_key)

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def _llm_section_33_interpretation(llm_client, lang_table_md: str, best_language: str, best_avg: float) -> str:
    if llm_client is None:
        return "_LLM interpretation skipped (LLM_URL/LLM not configured in root .env)._"
    system = (
        "You are an evaluator analyst. Interpret benchmark tables conservatively. "
        "Do not invent numbers. Keep the language factual and caveat-aware."
    )
    user = (
        "Interpret the following language-performance table from a multilingual LLM benchmark.\n"
        f"Observed top language in this run: {best_language} (avg={best_avg:.3f}).\n"
        "Keep it concise: 6-8 bullets total.\n"
        "Use only values present in the table. Do not invent rankings, causes, or metric values.\n"
        "If something cannot be inferred from the table, say so explicitly.\n"
        "You must explicitly restate these method caveats in plain language:\n"
        "- overall = weighted proxy score (relevance/factual/completeness/fluency/coherence)\n"
        "- factual_accuracy is NLI-based with xlm-roberta-large-xnli\n"
        "- fluency/coherence are zero-shot proxy signals\n"
        "- prompt_alignment and token_efficiency do not contribute in context mode\n"
        "- multilingual deltas can be noisy; avoid over-claiming\n"
        "Also include one bullet explaining how Romanian can appear on top under these proxy metrics.\n"
        "Return markdown with exactly two short headings:\n"
        "### Quick Interpretation\n### Validation Checks\n"
        "Under Validation Checks, provide exactly 3 checks.\n\n"
        "Table:\n"
        f"{lang_table_md}\n"
    )
    try:
        return llm_client.ask(system, user, max_tokens=420, temperature=0.0)
    except Exception as e:
        return f"_LLM interpretation unavailable: {type(e).__name__}. Run with network/LLM access or use `--no-llm`._"


def _llm_section_34_interpretation(llm_client, metric_table_md: str) -> str:
    if llm_client is None:
        return "_LLM interpretation skipped (LLM_URL/LLM not configured in root .env)._"
    system = (
        "You are an evaluator analyst. Interpret metric-breakdown tables conservatively. "
        "Do not invent numbers and do not over-claim causality."
    )
    user = (
        "Interpret the following per-model metric breakdown from a multilingual LLM benchmark.\n"
        "Keep it concise: 6-8 bullets total.\n"
        "Use only values present in the table. Do not invent rankings, causes, or metric values.\n"
        "If something cannot be inferred from the table, say so explicitly.\n"
        "You must explicitly mention:\n"
        "- weighted proxy nature of overall score\n"
        "- NLI/zero-shot dependence and resulting limitations\n"
        "- why small model gaps should be treated cautiously\n"
        "Return markdown with exactly two short headings:\n"
        "### Metric Readout\n### Practical Caveats\n"
        "Under Practical Caveats, provide exactly 4 bullets.\n\n"
        "Table:\n"
        f"{metric_table_md}\n"
    )
    try:
        return llm_client.ask(system, user, max_tokens=420, temperature=0.0)
    except Exception as e:
        return f"_LLM interpretation unavailable: {type(e).__name__}. Run with network/LLM access or use `--no-llm`._"


def _llm_gpu_efficiency_interpretation(llm_client, gpu_table_md: str, must_cover_model: str | None = None) -> str:
    if llm_client is None:
        return "_LLM interpretation skipped (LLM_URL/LLM not configured in root .env)._"
    system = (
        "You are an inference-systems analyst. Interpret GPU efficiency tables conservatively, "
        "without inventing numbers."
    )
    cover_line = ""
    if must_cover_model:
        cover_line = (
            f"You MUST include one explicit bullet about `{must_cover_model}` "
            "(its efficiency vs quality/latency trade-off).\n"
        )
    user = (
        "Interpret this model-level GPU efficiency table from a context-evaluation run.\n"
        "Keep it concise: 6-8 bullets total.\n"
        "Use only values present in the table. Do not invent rankings, causes, or metric values.\n"
        "If something cannot be inferred from the table, say so explicitly.\n"
        "Focus on trade-offs across: gpu_util_mean, memory usage, avg_latency_ms, and avg_overall_quality.\n"
        "Explicitly answer whether any model appears computationally efficient but weaker on quality, or vice versa.\n"
        f"{cover_line}"
        "Return markdown with exactly two headings:\n"
        "### Efficiency Readout\n### Practical Interpretation\n"
        "Under Practical Interpretation, include exactly 3 bullets.\n\n"
        "Table:\n"
        f"{gpu_table_md}\n"
    )
    try:
        return llm_client.ask(system, user, max_tokens=700, temperature=0.0)
    except Exception as e:
        return f"_LLM interpretation unavailable: {type(e).__name__}. Run with network/LLM access or use `--no-llm`._"


def _sanitize_section_33_text(
    text: str,
    lang_perf_df: pd.DataFrame,
    model_labels: dict[str, str],
) -> str:
    if not text or lang_perf_df.empty or not model_labels:
        return text
    top_row = lang_perf_df.iloc[0]
    label_cols = [v for v in model_labels.values() if v in lang_perf_df.columns]
    if not label_cols:
        return text
    try:
        vals = [float(top_row[c]) for c in label_cols if pd.notna(top_row[c])]
    except Exception:
        return text
    if not vals:
        return text
    vmin = min(vals)
    vmax = max(vals)
    code = str(top_row.get("Code", "N/A"))

    # Guard against common over-claim: "all eight metrics above 0.800".
    if vmin < 0.8:
        pattern = re.compile(r"(?im)^- .*all .* (above|over) 0\.?8\d*.*$")
        replacement = (
            f"- Sanity check from table values: for top language `{code}`, model-level scores range "
            f"from **{vmin:.3f}** to **{vmax:.3f}**; therefore not all model scores are above 0.800."
        )
        if pattern.search(text):
            text = pattern.sub(replacement, text)
        else:
            text = text.rstrip() + "\n" + replacement
    return text


def _iter_run_dirs(gpu_runs_dir: Path) -> list[Path]:
    return sorted([p for p in gpu_runs_dir.iterdir() if p.is_dir()])


def _read_scores(scores_db: Path) -> pd.DataFrame:
    if not scores_db.exists():
        return pd.DataFrame(
            columns=[
                "model_name",
                "language",
                "question_id",
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
    with sqlite3.connect(scores_db) as con:
        return pd.read_sql_query(
            """
            SELECT
                model_name,
                language,
                question_id,
                relevance,
                factual_accuracy,
                completeness,
                fluency,
                coherence,
                prompt_alignment,
                token_efficiency,
                overall_quality
            FROM scores
            """,
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


def _norm_model_name(v: str | None) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _read_gpu_metrics(gpu_csv: Path, run_name: str) -> pd.DataFrame:
    if not gpu_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(gpu_csv)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if any(c not in df.columns for c in GPU_REQUIRED_COLS):
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["util_gpu_pct"] = pd.to_numeric(df["util_gpu_pct"], errors="coerce")
    df["mem_used_mb"] = pd.to_numeric(df["mem_used_mb"], errors="coerce")
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df["phase"] = df["phase"].fillna("unknown").astype(str)
    df["run_name"] = run_name
    if "model_name" in df.columns:
        df["model_name_norm"] = df["model_name"].map(_norm_model_name).replace("", np.nan)
    else:
        df["model_name_norm"] = np.nan
    return df


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


def build_report(gpu: str, run_dirs: list[Path], generation_profile: dict, llm_client=None) -> str:
    all_scores: list[pd.DataFrame] = []
    all_results: list[pd.DataFrame] = []
    all_token_rows: list[pd.DataFrame] = []
    all_gpu_rows: list[pd.DataFrame] = []
    scores_xlsx_rows = 0
    results_xlsx_rows = 0
    results_by_model_all_rows = 0

    for run_dir in run_dirs:
        scores_db = run_dir / "scores" / "evaluation_scores_euf_context.db"
        results_db = run_dir / "raw" / "evaluation_results_euf_context.db"
        all_scores.append(_read_scores(scores_db))
        all_results.append(_read_results(results_db))
        all_gpu_rows.append(_read_gpu_metrics(run_dir / "logs" / "gpu_metrics.csv", run_dir.name))
        token_detail_csv = run_dir / "insights" / "data" / "token_budget_response_details.csv"
        if token_detail_csv.exists():
            try:
                all_token_rows.append(pd.read_csv(token_detail_csv))
            except Exception:
                pass
        scores_xlsx_rows += _read_xlsx_rows(run_dir / "scores" / "evaluation_scores_euf_context.xlsx", 0)
        results_xlsx_rows += _read_xlsx_rows(run_dir / "raw" / "evaluation_results_euf_context.xlsx", 0)
        results_by_model_all_rows += _read_xlsx_rows(
            run_dir / "raw" / "evaluation_results_euf_context_by_model.xlsx",
            "all_results",
        )

    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    token_rows = pd.concat(all_token_rows, ignore_index=True) if all_token_rows else pd.DataFrame()
    gpu_rows = pd.concat(all_gpu_rows, ignore_index=True) if all_gpu_rows else pd.DataFrame()

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
    fastest_model_quality: float | None = None
    fastest_good_model_name: str | None = None
    fastest_good_latency: float | None = None
    quality_threshold: float | None = None
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
            if not scores.empty and {"model_name", "overall_quality"}.issubset(scores.columns):
                qmap = scores.groupby("model_name")["overall_quality"].mean().to_dict()
                if fastest_model_name in qmap:
                    fastest_model_quality = float(qmap[fastest_model_name])

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
    if quality_threshold is not None:
        lines.append(f"- Quality qualification threshold (median avg_overall): **{quality_threshold:.3f}**")
    if fastest_model_name and fastest_model_quality is not None and quality_threshold is not None:
        qualifies = fastest_model_quality >= quality_threshold
        status = "qualifies" if qualifies else "does not qualify"
        lines.append(
            f"- Fastest model quality check: **{fastest_model_name} avg_overall={fastest_model_quality:.3f} "
            f"({status} vs threshold {quality_threshold:.3f})**"
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
    if quality_threshold is not None:
        lines.append(
            f"_Quality-qualified set uses models with average overall quality >= median model quality "
            f"(threshold = {quality_threshold:.3f})._"
        )
    else:
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
    lines.append("## Token Budget Analysis (Current Context Evaluation)")
    lines.append("")
    lines.append(
        "This section is computed from `insights/data/token_budget_response_details.csv` across discovered runs."
    )
    if token_rows.empty:
        lines.append("")
        lines.append("_No token budget detail files found. Run `insights/generate_context_token_budget.py` first._")
        lines.append("")
    else:
        # Normalize numeric columns for stable aggregation.
        for c in [
            "input_tokens",
            "response_tokens",
            "total_tokens",
            "max_model_len",
            "remaining_output_tokens",
            "effective_output_cap",
        ]:
            if c in token_rows.columns:
                token_rows[c] = pd.to_numeric(token_rows[c], errors="coerce")

        token_source_df = pd.DataFrame(columns=["token_count_source", "rows"])
        if "token_count_source" in token_rows.columns:
            token_source_df = (
                token_rows.groupby("token_count_source")
                .size()
                .reset_index(name="rows")
                .sort_values("rows", ascending=False)
            )

        lines.append("### Token Count Source")
        lines.append("")
        lines.append(_to_md_table(token_source_df) if not token_source_df.empty else "_No data_")
        lines.append("")

        # A) Base question profile
        a_df = pd.DataFrame()
        if {"base_question", "input_tokens", "response_tokens", "total_tokens"}.issubset(token_rows.columns):
            a_df = (
                token_rows.groupby("base_question")
                .agg(
                    input_tokens_mean=("input_tokens", "mean"),
                    input_tokens_min=("input_tokens", "min"),
                    input_tokens_max=("input_tokens", "max"),
                    response_tokens_mean=("response_tokens", "mean"),
                    response_tokens_p90=("response_tokens", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
                    response_tokens_max=("response_tokens", "max"),
                    total_tokens_mean=("total_tokens", "mean"),
                    total_tokens_max=("total_tokens", "max"),
                )
                .reset_index()
                .sort_values("base_question")
            )
        lines.append("### A) Input/Output Token Profile by Base Question (from real responses)")
        lines.append("")
        lines.append(
            _to_md_table_fmt(
                a_df,
                float_cols=[
                    "input_tokens_mean",
                    "response_tokens_mean",
                    "response_tokens_p90",
                    "total_tokens_mean",
                ],
                ndigits=1,
            ) if not a_df.empty else "_No data_"
        )
        lines.append("")

        # B) Per-model output budget
        b_df = pd.DataFrame()
        needed_b = {
            "model_name",
            "max_model_len",
            "max_model_len_source",
            "input_tokens",
            "response_tokens",
            "remaining_output_tokens",
        }
        if needed_b.issubset(token_rows.columns):
            b_df = (
                token_rows.groupby("model_name")
                .agg(
                    max_model_len=("max_model_len", "max"),
                    max_model_len_source=("max_model_len_source", "first"),
                    input_tokens_mean=("input_tokens", "mean"),
                    response_tokens_mean=("response_tokens", "mean"),
                    response_tokens_p90=("response_tokens", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
                    response_tokens_max=("response_tokens", "max"),
                    remaining_output_tokens_min=("remaining_output_tokens", "min"),
                    remaining_output_tokens_max=("remaining_output_tokens", "max"),
                )
                .reset_index()
                .sort_values("model_name")
            )
        lines.append("### B) Per-Model Output Budget Using Config Max Context")
        lines.append("")
        lines.append(
            _to_md_table_fmt(
                b_df,
                float_cols=[
                    "input_tokens_mean",
                    "response_tokens_mean",
                    "response_tokens_p90",
                ],
                ndigits=1,
            ) if not b_df.empty else "_No data_"
        )
        lines.append("")
        if not b_df.empty and "remaining_output_tokens_min" in b_df.columns:
            overflow_df = b_df[b_df["remaining_output_tokens_min"] < 0].copy()
            if not overflow_df.empty:
                lines.append("#### Token Budget Overflow Warning")
                lines.append("")
                lines.append(
                    "The following models have **negative remaining output budget** for observed inputs. "
                    "This means observed prompt+context tokens exceeded configured `max_model_len` and runtime "
                    "will rely on truncation/compaction behavior."
                )
                lines.append("")
                lines.append(
                    _to_md_table_fmt(
                        overflow_df[
                            [
                                "model_name",
                                "max_model_len",
                                "input_tokens_mean",
                                "remaining_output_tokens_min",
                                "remaining_output_tokens_max",
                            ]
                        ],
                        float_cols=["input_tokens_mean"],
                        ndigits=1,
                    )
                )
                lines.append("")

        # C) Per-language x per-question
        c_df = pd.DataFrame()
        if {"language", "base_question", "input_tokens", "response_tokens", "total_tokens"}.issubset(token_rows.columns):
            c_df = (
                token_rows.groupby(["language", "base_question"])
                .agg(
                    input_tokens=("input_tokens", "mean"),
                    response_tokens_mean=("response_tokens", "mean"),
                    response_tokens_p90=("response_tokens", lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) else np.nan),
                    response_tokens_max=("response_tokens", "max"),
                    total_tokens_mean=("total_tokens", "mean"),
                    total_tokens_max=("total_tokens", "max"),
                )
                .reset_index()
                .sort_values(["language", "base_question"])
            )
        lines.append("### C) Per-Language x Per-Question (Input + Response + Total)")
        lines.append("")
        lines.append(
            _to_md_table_fmt(
                c_df,
                float_cols=[
                    "input_tokens",
                    "response_tokens_mean",
                    "response_tokens_p90",
                    "total_tokens_mean",
                ],
                ndigits=1,
            ) if not c_df.empty else "_No data_"
        )
        lines.append("")

    lines.append("## Metric Methodology and Caveats")
    lines.append("")
    lines.append(
        "- Scoring is executed with the **context** profile in `gpu_runtime/evaluate_context_results.py` "
        "via `ResponseEvaluator(metrics_profile=\"context\")`."
    )
    lines.append("- Context profile weights used in overall quality:")
    lines.append("  - `relevance`: 0.30")
    lines.append("  - `factual_accuracy`: 0.30")
    lines.append("  - `completeness`: 0.20")
    lines.append("  - `fluency`: 0.15")
    lines.append("  - `coherence`: 0.05")
    lines.append("  - `prompt_alignment`: 0.00 (not contributing in context mode)")
    lines.append("  - `token_efficiency`: 0.00 (disabled in context mode)")
    lines.append(
        "- `factual_accuracy` is NLI-based using **`joeddav/xlm-roberta-large-xnli`** "
        "on response-vs-context premise/hypothesis pairs."
    )
    lines.append(
        "- `fluency` and `coherence` are **zero-shot classification** scores "
        "using the same multilingual XLM-RoBERTa model with labels "
        "(`fluent`/`not fluent`, `coherent`/`incoherent`)."
    )
    lines.append(
        "- Zero-shot and NLI scores are proxy metrics: good for scalable comparison, "
        "but they are not a substitute for expert human judgment."
    )
    lines.append(
        "- Multilingual reliability is generally good but **uneven across languages**, "
        "especially for lower-resource languages. Treat small language deltas cautiously."
    )
    lines.append(
        "- The warning about unused `pooler` weights for XLM-R checkpoints is expected in this pipeline "
        "and does not indicate a scoring failure."
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
    lines.append("### How To Read Model Ranking")
    lines.append("")
    lines.append("- `avg_overall`: average overall quality score for that model across all evaluated responses.")
    lines.append("- `std_overall`: spread/consistency of scores.")
    lines.append("  Lower `std_overall` means more stable performance; higher means more variability across language/question/run.")
    lines.append("- `p10`: 10th percentile score.")
    lines.append("  About 10% of responses are at or below this value (lower-tail/worse-case tendency).")
    lines.append("- `p90`: 90th percentile score.")
    lines.append("  About 90% of responses are at or below this value; top ~10% are above it (upper-tail/best-case tendency).")
    if not model_ranking.empty:
        ex = model_ranking.iloc[0]
        lines.append(
            f"- Example (`{ex['model_name']}`): avg={float(ex['avg_overall']):.3f}, "
            f"std={float(ex['std_overall']):.3f}, p10={float(ex['p10']):.3f}, p90={float(ex['p90']):.3f}. "
            "Interpretation: typical quality is near the average, weaker outputs cluster around p10, and strong outputs reach around p90."
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
    lang_perf_df = pd.DataFrame()
    lang_perf_md = "_No data_"
    lang_best_code = ""
    lang_best_avg = float("nan")
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
        lang_perf_md = _to_md_table_fmt(lang_perf_df, float_cols=float_cols, ndigits=3)
        lines.append(lang_perf_md)
        if not lang_perf_df.empty:
            lang_best_code = str(lang_perf_df.iloc[0]["Code"])
            try:
                lang_best_avg = float(lang_perf_df.iloc[0]["Avg"])
            except Exception:
                lang_best_avg = float("nan")
    else:
        lines.append(lang_perf_md)
    lines.append("")
    lines.append("#### 3.3 Interpretation")
    lines.append("")
    lines.append("_Note: AI-generated interpretation; may contain errors. Use tables above as source of truth._")
    lines.append("")
    llm_33 = _llm_section_33_interpretation(
        llm_client=llm_client,
        lang_table_md=lang_perf_md,
        best_language=lang_best_code or "N/A",
        best_avg=lang_best_avg if pd.notna(lang_best_avg) else 0.0,
    )
    llm_33 = _sanitize_section_33_text(
        llm_33,
        lang_perf_df=lang_perf_df,
        model_labels=model_labels,
    )
    lines.append(llm_33)
    lines.append("")
    lines.append("### 3.4 Detailed Metric Breakdown (All Evaluated Models)")
    lines.append("")
    metric_breakdown_md = "_No data_"
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
        metric_breakdown_md = _to_md_table_fmt(
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
    else:
        lines.append("_No data_")
    lines.append("")
    lines.append("#### 3.4 Interpretation")
    lines.append("")
    lines.append("_Note: AI-generated interpretation; may contain errors. Use tables above as source of truth._")
    lines.append("")
    lines.append(_llm_section_34_interpretation(llm_client=llm_client, metric_table_md=metric_breakdown_md))
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
    lines.append("## GPU Efficiency Summary (Merged)")
    lines.append("")
    lines.append(
        "This section merges runtime GPU telemetry from `logs/gpu_metrics.csv` across all discovered runs "
        f"under `results/runs/{gpu}`."
    )
    lines.append("")
    if gpu_rows.empty:
        lines.append("_No valid GPU metrics found. Generate/collect `logs/gpu_metrics.csv` per run to populate this section._")
        lines.append("")
    else:
        g = gpu_rows.dropna(subset=["timestamp"]).copy()
        if g.empty:
            lines.append("_GPU metrics files were present but timestamps were invalid._")
            lines.append("")
        else:
            start = g["timestamp"].min()
            end = g["timestamp"].max()
            duration_h = (end - start).total_seconds() / 3600.0
            util_mean = float(g["util_gpu_pct"].mean())
            util_p10 = float(g["util_gpu_pct"].quantile(0.10))
            util_p90 = float(g["util_gpu_pct"].quantile(0.90))
            mem_mean_gb = float(g["mem_used_mb"].mean() / 1024.0)
            mem_p10_gb = float(g["mem_used_mb"].quantile(0.10) / 1024.0)
            mem_p90_gb = float(g["mem_used_mb"].quantile(0.90) / 1024.0)
            mem_max_gb = float(g["mem_used_mb"].max() / 1024.0)
            temp_max_c = float(g["temp_c"].max())

            lines.append("### Core GPU Statistics")
            lines.append("")
            lines.append(f"- Samples: **{len(g)}**")
            lines.append(f"- Time range: **{start} -> {end}**")
            lines.append(f"- Duration: **{duration_h:.2f} hours**")
            lines.append(f"- Mean GPU utilization: **{util_mean:.2f}%**")
            lines.append(f"- P10 GPU utilization: **{util_p10:.2f}%**")
            lines.append(f"- P90 GPU utilization: **{util_p90:.2f}%**")
            lines.append(f"- Mean GPU memory used: **{mem_mean_gb:.2f} GB**")
            lines.append(f"- P10 GPU memory used: **{mem_p10_gb:.2f} GB**")
            lines.append(f"- P90 GPU memory used: **{mem_p90_gb:.2f} GB**")
            lines.append(f"- Max GPU memory used: **{mem_max_gb:.2f} GB**")
            lines.append(f"- Max GPU temperature: **{temp_max_c:.1f} C**")
            lines.append("")

            lines.append("### Phase Breakdown")
            lines.append("")
            phase_df = (
                g["phase"].value_counts(dropna=False).rename_axis("phase").reset_index(name="samples")
            )
            phase_df["pct"] = (phase_df["samples"] / phase_df["samples"].sum() * 100.0).round(2)
            lines.append(_to_md_table_fmt(phase_df, float_cols=["pct"], ndigits=2))
            lines.append("")

            lines.append("### Model-Level GPU Efficiency")
            lines.append("")
            mg = g[g["model_name_norm"].notna() & (g["model_name_norm"] != "")].copy()
            if mg.empty:
                lines.append("_No model-level GPU metrics available in telemetry files._")
                lines.append("")
            else:
                model_gpu = (
                    mg.groupby("model_name_norm", as_index=False)
                    .agg(
                        samples=("timestamp", "count"),
                        start_ts=("timestamp", "min"),
                        end_ts=("timestamp", "max"),
                        gpu_util_p10=("util_gpu_pct", lambda x: np.nanquantile(x, 0.10)),
                        gpu_util_mean=("util_gpu_pct", "mean"),
                        gpu_util_p90=("util_gpu_pct", lambda x: np.nanquantile(x, 0.90)),
                        mem_used_p10_gb=("mem_used_mb", lambda x: np.nanquantile(x, 0.10) / 1024.0),
                        mem_used_mean_gb=("mem_used_mb", "mean"),
                        mem_used_p90_gb=("mem_used_mb", lambda x: np.nanquantile(x, 0.90) / 1024.0),
                        mem_used_max_gb=("mem_used_mb", "max"),
                        temp_max_c=("temp_c", "max"),
                    )
                )
                model_gpu["mem_used_mean_gb"] = model_gpu["mem_used_mean_gb"] / 1024.0
                model_gpu["mem_used_max_gb"] = model_gpu["mem_used_max_gb"] / 1024.0
                model_gpu["duration_min"] = (
                    (model_gpu["end_ts"] - model_gpu["start_ts"]).dt.total_seconds() / 60.0
                )

                latency_lookup = pd.DataFrame()
                if not results.empty and {"model_name", "latency_ms"}.issubset(results.columns):
                    latency_lookup = (
                        results.dropna(subset=["latency_ms"])
                        .groupby("model_name", as_index=False)
                        .agg(avg_latency_ms=("latency_ms", "mean"))
                    )
                    latency_lookup["model_name_norm"] = latency_lookup["model_name"].map(_norm_model_name)

                quality_lookup = pd.DataFrame()
                if not scores.empty and {"model_name", "overall_quality"}.issubset(scores.columns):
                    quality_lookup = (
                        scores.groupby("model_name", as_index=False)
                        .agg(avg_overall_quality=("overall_quality", "mean"))
                    )
                    quality_lookup["model_name_norm"] = quality_lookup["model_name"].map(_norm_model_name)

                model_eff = model_gpu.copy()
                if not latency_lookup.empty:
                    model_eff = model_eff.merge(
                        latency_lookup[["model_name_norm", "model_name", "avg_latency_ms"]],
                        on="model_name_norm",
                        how="left",
                    )
                if not quality_lookup.empty:
                    model_eff = model_eff.merge(
                        quality_lookup[["model_name_norm", "avg_overall_quality"]],
                        on="model_name_norm",
                        how="left",
                    )
                if "model_name" not in model_eff.columns:
                    model_eff["model_name"] = model_eff["model_name_norm"]
                else:
                    model_eff["model_name"] = model_eff["model_name"].fillna(model_eff["model_name_norm"])
                model_eff["quality_per_second"] = np.where(
                    (model_eff["avg_latency_ms"].notna())
                    & (model_eff["avg_latency_ms"] > 0)
                    & (model_eff["avg_overall_quality"].notna()),
                    model_eff["avg_overall_quality"] / (model_eff["avg_latency_ms"] / 1000.0),
                    np.nan,
                )

                model_eff = model_eff.sort_values("gpu_util_mean", ascending=False)
                model_eff = model_eff[
                    [
                        "model_name",
                        "duration_min",
                        "gpu_util_p10",
                        "gpu_util_mean",
                        "gpu_util_p90",
                        "mem_used_p10_gb",
                        "mem_used_mean_gb",
                        "mem_used_p90_gb",
                        "mem_used_max_gb",
                        "temp_max_c",
                        "avg_latency_ms",
                        "avg_overall_quality",
                        "quality_per_second",
                    ]
                ]
                gpu_eff_md = _to_md_table_fmt(
                    model_eff,
                    float_cols=[
                        "duration_min",
                        "gpu_util_p10",
                        "gpu_util_mean",
                        "gpu_util_p90",
                        "mem_used_p10_gb",
                        "mem_used_mean_gb",
                        "mem_used_p90_gb",
                        "mem_used_max_gb",
                        "temp_max_c",
                        "avg_latency_ms",
                        "avg_overall_quality",
                        "quality_per_second",
                    ],
                    ndigits=3,
                )
                lines.append(gpu_eff_md)
                must_cover_model = None
                qwen_matches = [
                    str(m)
                    for m in model_eff["model_name"].dropna().astype(str).tolist()
                    if "qwen3-30b" in m.lower()
                ]
                if qwen_matches:
                    must_cover_model = qwen_matches[0]
                lines.append("")
                lines.append("#### Model-Level GPU Efficiency Interpretation")
                lines.append("")
                lines.append("_Note: AI-generated interpretation; may contain errors. Use tables above as source of truth._")
                lines.append("")
                lines.append(
                    _llm_gpu_efficiency_interpretation(
                        llm_client=llm_client,
                        gpu_table_md=gpu_eff_md,
                        must_cover_model=must_cover_model,
                    )
                )
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
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-assisted interpretation blocks for sections 3.3 and 3.4.",
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
    llm_client = None if args.no_llm else _LLMClient.from_repo_env(ROOT)
    report = build_report(
        gpu=gpu,
        run_dirs=run_dirs,
        generation_profile=generation_profile,
        llm_client=llm_client,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
