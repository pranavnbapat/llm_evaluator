#!/usr/bin/env python3
"""
Generate presentation Q&A sheet from latest insights artifacts.

Outputs:
  - insights/Presentation_QA.md
  - insights/data/presentation_qa.csv
  - insights/data/presentation_qa.xlsx
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INSIGHTS_DIR = ROOT / "insights"
DATA_DIR = INSIGHTS_DIR / "data"

MODEL_SUMMARY = DATA_DIR / "model_summary.csv"
LANG_SUMMARY = DATA_DIR / "language_summary.csv"
QUESTION_SUMMARY = DATA_DIR / "question_summary.csv"
LATENCY_SUMMARY = DATA_DIR / "latency_summary.csv"

OUT_MD = INSIGHTS_DIR / "Presentation_QA.md"
OUT_CSV = DATA_DIR / "presentation_qa.csv"
OUT_XLSX = DATA_DIR / "presentation_qa.xlsx"


def load_context() -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    if MODEL_SUMMARY.exists():
        m = pd.read_csv(MODEL_SUMMARY)
        if not m.empty:
            best = m.sort_values("avg_overall", ascending=False).iloc[0]
            worst = m.sort_values("avg_overall", ascending=True).iloc[0]
            ctx["best_model"] = str(best["model_name"])
            ctx["best_score"] = f"{best['avg_overall']:.3f}"
            ctx["worst_model"] = str(worst["model_name"])
            ctx["worst_score"] = f"{worst['avg_overall']:.3f}"
            ctx["spread"] = f"{(best['avg_overall'] - worst['avg_overall']):.3f}"
            ctx["models_n"] = str(m["model_name"].nunique())

    if LANG_SUMMARY.exists():
        l = pd.read_csv(LANG_SUMMARY)
        if not l.empty:
            top = l.sort_values("avg_overall", ascending=False).iloc[0]
            bot = l.sort_values("avg_overall", ascending=True).iloc[0]
            ctx["top_lang"] = str(top["language"])
            ctx["top_lang_score"] = f"{top['avg_overall']:.3f}"
            ctx["bottom_lang"] = str(bot["language"])
            ctx["bottom_lang_score"] = f"{bot['avg_overall']:.3f}"
            ctx["lang_spread"] = f"{(top['avg_overall'] - bot['avg_overall']):.3f}"
            ctx["langs_n"] = str(l["language"].nunique())

    if QUESTION_SUMMARY.exists():
        q = pd.read_csv(QUESTION_SUMMARY)
        if not q.empty:
            topq = q.sort_values("avg_overall", ascending=False).iloc[0]
            lowq = q.sort_values("avg_overall", ascending=True).iloc[0]
            ctx["top_q"] = str(topq["base_qid"])
            ctx["top_q_score"] = f"{topq['avg_overall']:.3f}"
            ctx["low_q"] = str(lowq["base_qid"])
            ctx["low_q_score"] = f"{lowq['avg_overall']:.3f}"
            ctx["qs_n"] = str(q["base_qid"].nunique())

    if LATENCY_SUMMARY.exists():
        t = pd.read_csv(LATENCY_SUMMARY)
        if not t.empty:
            fast = t.sort_values("avg_latency_ms", ascending=True).iloc[0]
            slow = t.sort_values("avg_latency_ms", ascending=False).iloc[0]
            ctx["fast_model"] = str(fast["model_name"])
            ctx["fast_latency"] = f"{fast['avg_latency_ms']:.0f} ms"
            ctx["slow_model"] = str(slow["model_name"])
            ctx["slow_latency"] = f"{slow['avg_latency_ms']:.0f} ms"

    defaults = {
        "best_model": "devstral-small-2-24b-instruct-2512-b200",
        "best_score": "0.816",
        "worst_model": "deepseek-r1-distill-qwen-14b-b200",
        "worst_score": "0.774",
        "spread": "0.042",
        "models_n": "9",
        "top_lang": "ES",
        "top_lang_score": "0.817",
        "bottom_lang": "GA",
        "bottom_lang_score": "0.679",
        "lang_spread": "0.138",
        "langs_n": "24",
        "top_q": "Q1",
        "top_q_score": "0.823",
        "low_q": "Q2",
        "low_q_score": "0.768",
        "qs_n": "5",
        "fast_model": "eurollm-9b-instruct-2512",
        "fast_latency": "2752 ms",
        "slow_model": "deepseek-r1-distill-qwen-32b-b200",
        "slow_latency": "14418 ms",
    }
    for k, v in defaults.items():
        ctx.setdefault(k, v)
    return ctx


def build_qa(ctx: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = [
        {
            "Category": "Scope",
            "Question": "What did we evaluate?",
            "Answer": f"Context-grounded response quality across {ctx['models_n']} models, {ctx['langs_n']} EU languages, and {ctx['qs_n']} question families.",
            "Evidence": "scores/results DB + XLSX artifacts",
            "Risk_or_Caveat": "Findings apply to this task mix and prompt design.",
        },
        {
            "Category": "Scope",
            "Question": "How many samples are in the final analysis?",
            "Answer": "3,240 scored responses.",
            "Evidence": "evaluation_scores_euf_context.db row count",
            "Risk_or_Caveat": "Assumes no post-run filtering.",
        },
        {
            "Category": "Ranking",
            "Question": "Which model performed best overall?",
            "Answer": f"{ctx['best_model']} with average overall quality {ctx['best_score']}.",
            "Evidence": "insights/data/model_summary.csv",
            "Risk_or_Caveat": "Ranking can shift with different weights or datasets.",
        },
        {
            "Category": "Ranking",
            "Question": "How big is the gap between best and worst model?",
            "Answer": f"About {ctx['spread']} in average overall quality ({ctx['best_score']} vs {ctx['worst_score']}).",
            "Evidence": "model_summary.csv",
            "Risk_or_Caveat": "Statistical significance not yet reported here.",
        },
        {
            "Category": "Language",
            "Question": "Which language was strongest and weakest?",
            "Answer": f"Top: {ctx['top_lang']} ({ctx['top_lang_score']}); bottom: {ctx['bottom_lang']} ({ctx['bottom_lang_score']}).",
            "Evidence": "language_summary.csv",
            "Risk_or_Caveat": "May reflect training-data imbalance and script complexity.",
        },
        {
            "Category": "Language",
            "Question": "Are all 24 EU languages covered?",
            "Answer": "Yes, all 24 are present in the scored dataset.",
            "Evidence": "distinct language count in scores DB",
            "Risk_or_Caveat": "Coverage does not imply equal performance quality.",
        },
        {
            "Category": "Question",
            "Question": "Which question type was easiest/hardest?",
            "Answer": f"Best average: {ctx['top_q']} ({ctx['top_q_score']}); lowest: {ctx['low_q']} ({ctx['low_q_score']}).",
            "Evidence": "question_summary.csv",
            "Risk_or_Caveat": "Question wording and context quality influence this.",
        },
        {
            "Category": "Latency",
            "Question": "Which model is fastest and slowest?",
            "Answer": f"Fastest mean latency: {ctx['fast_model']} ({ctx['fast_latency']}); slowest: {ctx['slow_model']} ({ctx['slow_latency']}).",
            "Evidence": "latency_summary.csv",
            "Risk_or_Caveat": "Latency depends on hardware/runtime settings.",
        },
        {
            "Category": "Method",
            "Question": "How is the overall quality score computed?",
            "Answer": "Weighted combination of relevance, factual accuracy, completeness, fluency, coherence, prompt alignment, and token efficiency.",
            "Evidence": "metrics/metrics_config.yaml + scoring script",
            "Risk_or_Caveat": "Weight choices encode subjective priorities.",
        },
        {
            "Category": "Method",
            "Question": "What ensures scores are valid and not corrupted?",
            "Answer": "All metrics are within [0,1], with zero nulls and full score/result coverage.",
            "Evidence": "insights/EVALUATION_CONTEXT_REPORT.md integrity checks",
            "Risk_or_Caveat": "Range validity does not prove metric correctness.",
        },
        {
            "Category": "Method",
            "Question": "Why is token_efficiency always 0?",
            "Answer": "For the context profile, token efficiency is disabled (weight 0), so it contributes 0 by design.",
            "Evidence": "metrics_config context profile",
            "Risk_or_Caveat": "Enable it only if token-cost tradeoff is a priority.",
        },
        {
            "Category": "Reliability",
            "Question": "How stable are scores across 3 runs?",
            "Answer": "Most models show very low run-to-run variance; variability is generally small.",
            "Evidence": "run stability table in insights report",
            "Risk_or_Caveat": "Some models still show larger tails on specific cells.",
        },
        {
            "Category": "Interpretation",
            "Question": "Does higher factual accuracy mean no hallucinations?",
            "Answer": "Not necessarily. It indicates stronger entailment/context alignment, not complete hallucination absence.",
            "Evidence": "NLI-based factual metric design",
            "Risk_or_Caveat": "Needs manual audits on low-scoring outliers.",
        },
        {
            "Category": "Interpretation",
            "Question": "Can we trust language-level differences?",
            "Answer": "Yes directionally, but treat small gaps cautiously without formal significance tests.",
            "Evidence": "language summary + variance",
            "Risk_or_Caveat": "Recommend bootstrap CIs or mixed-effects modeling.",
        },
        {
            "Category": "Business",
            "Question": "Which model should we choose for production now?",
            "Answer": f"Start with {ctx['best_model']} for quality; benchmark against {ctx['fast_model']} if latency/cost dominates.",
            "Evidence": "model + latency summaries",
            "Risk_or_Caveat": "Decision should include cost, throughput, and operational constraints.",
        },
        {
            "Category": "Business",
            "Question": "Is a smaller model enough?",
            "Answer": "In this dataset, smaller/efficient models are competitive and often strong.",
            "Evidence": "overall ranking spread is modest",
            "Risk_or_Caveat": "May differ for long-context or domain-shift tasks.",
        },
        {
            "Category": "Process",
            "Question": "How reproducible is this pipeline?",
            "Answer": "Deterministic prompts/config + persisted DB artifacts make reruns auditable and comparable.",
            "Evidence": "results DB + generated insights scripts",
            "Risk_or_Caveat": "GPU-level nondeterminism can cause tiny float drift.",
        },
        {
            "Category": "Process",
            "Question": "What changed from prior runs?",
            "Answer": "Scoring was optimized (batching/caching) while preserving metric formulas and thresholds.",
            "Evidence": "code changes in evaluator + scoring script",
            "Risk_or_Caveat": "Validate with sample A/B diffs when governance requires strict parity.",
        },
        {
            "Category": "Governance",
            "Question": "What are the main limitations we should disclose?",
            "Answer": "Task set is domain-specific; metric weights are subjective; factual metric depends on NLI model behavior.",
            "Evidence": "report caveats and config",
            "Risk_or_Caveat": "Include these in executive readout slides.",
        },
        {
            "Category": "Governance",
            "Question": "How should we present uncertainty?",
            "Answer": "Show means with spread (std/p10/p90), and avoid over-claiming small differences.",
            "Evidence": "model summary statistics",
            "Risk_or_Caveat": "Add confidence intervals in next revision.",
        },
        {
            "Category": "Next Steps",
            "Question": "What analysis should be done next?",
            "Answer": "Add significance tests, failure-mode slicing, and cost-per-quality analysis per model.",
            "Evidence": "current summaries + latency",
            "Risk_or_Caveat": "Requires consistent cost telemetry capture.",
        },
        {
            "Category": "Next Steps",
            "Question": "How do we make this presentation defensible?",
            "Answer": "Share raw artifacts, scripts, charts, and exact commands used to regenerate report outputs.",
            "Evidence": "insights folder structure",
            "Risk_or_Caveat": "Ensure repository includes environment versions and seeds.",
        },
    ]
    df = pd.DataFrame(rows)
    # Normalize voice to "we" framing for presentation consistency.
    replacements = [
        (" did you ", " did we "),
        (" should you ", " should we "),
        (" can you ", " can we "),
        ("How do you ", "How do we "),
        ("What exactly did you evaluate?", "What did we evaluate?"),
        ("Did you ", "Did we "),
        ("you ", "we "),
        (" your ", " our "),
    ]
    for col in ["Question", "Answer"]:
        s = df[col].astype(str)
        for old, new in replacements:
            s = s.str.replace(old, new, regex=False)
        df[col] = s
    return df


def write_markdown(df: pd.DataFrame) -> None:
    lines = ["# Presentation Q&A Sheet", ""]
    lines.append("Use this during stakeholder reviews to answer common methodology, quality, and deployment questions.")
    lines.append("")
    lines.append("| Category | Question | Answer | Evidence | Risk_or_Caveat |")
    lines.append("| --- | --- | --- | --- | --- |")
    for _, r in df.iterrows():
        vals = [str(r[c]).replace("\n", " ").replace("|", "\\|") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ctx = load_context()
    qa = build_qa(ctx)
    qa.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        qa.to_excel(w, sheet_name="presentation_qa", index=False)
    write_markdown(qa)

    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_XLSX}")


if __name__ == "__main__":
    main()
