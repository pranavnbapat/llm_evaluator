#!/usr/bin/env python3
"""
Generate charts and tabular summaries for context-evaluation outputs.

Single-run mode:
  python insights/generate_context_charts.py --run-dir <run_dir>

Bulk mode (default):
  python insights/generate_context_charts.py
This scans results/runs/*/* and generates only missing chart/data artifacts.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _base_question_id(qid: str) -> str:
    if isinstance(qid, str) and "_" in qid:
        return qid.split("_", 1)[0]
    return str(qid)


def _read_table(db: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def build_summaries(scores: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()
    results = results.copy()
    scores["base_qid"] = scores["question_id"].map(_base_question_id)
    results["base_qid"] = results["question_id"].map(_base_question_id)

    model_summary = (
        scores.groupby("model_name")
        .agg(
            n=("id", "count"),
            avg_overall=("overall_quality", "mean"),
            std_overall=("overall_quality", "std"),
        )
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )

    language_summary = (
        scores.groupby("language")
        .agg(
            n=("id", "count"),
            avg_overall=("overall_quality", "mean"),
            std_overall=("overall_quality", "std"),
        )
        .sort_values("avg_overall", ascending=False)
        .reset_index()
    )

    question_summary = (
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

    # Keep only completed responses with valid latency for latency-based summaries/charts.
    if "response" in results.columns:
        results = results[results["response"].notna()].copy()
    if "latency_ms" in results.columns:
        results["latency_ms"] = pd.to_numeric(results["latency_ms"], errors="coerce")
        results = results[results["latency_ms"].notna()].copy()

    latency_summary = (
        results.groupby("model_name")
        .agg(
            n=("id", "count"),
            avg_latency_ms=("latency_ms", "mean"),
            p90_latency_ms=("latency_ms", lambda x: np.percentile(x, 90)),
        )
        .reset_index()
    )

    return model_summary, language_summary, question_summary, latency_summary


def plot_model_overall_bar(model_summary: pd.DataFrame, out: Path) -> None:
    df = model_summary.copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(df["model_name"], df["avg_overall"], color="#2a9d8f")
    ax.set_title("Average Overall Quality by Model")
    ax.set_ylabel("Overall Quality (0-1)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    for b, val in zip(bars, df["avg_overall"]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    _save_fig(out)


def plot_metric_heatmap(scores: pd.DataFrame, out: Path) -> None:
    metric_by_model = scores.groupby("model_name")[SCORE_COLS].mean().sort_values("overall_quality", ascending=False)
    if metric_by_model.empty:
        return
    data = metric_by_model.values
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title("Metric Means by Model")
    ax.set_xticks(np.arange(len(SCORE_COLS)))
    ax.set_xticklabels(SCORE_COLS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(metric_by_model.index)))
    ax.set_yticklabels(metric_by_model.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save_fig(out)


def plot_language_overall(language_summary: pd.DataFrame, out: Path) -> None:
    df = language_summary.copy().sort_values("avg_overall", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["language"], df["avg_overall"], color="#457b9d")
    ax.set_title("Average Overall Quality by Language")
    ax.set_ylabel("Overall Quality (0-1)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    _save_fig(out)


def plot_question_overall(question_summary: pd.DataFrame, out: Path) -> None:
    df = question_summary.copy().sort_values("base_qid")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(df["base_qid"], df["avg_overall"], color="#e76f51")
    ax.set_title("Average Overall Quality by Question Family")
    ax.set_ylabel("Overall Quality (0-1)")
    ax.set_ylim(0, 1)
    _save_fig(out)


def plot_latency_vs_quality(model_summary: pd.DataFrame, latency_summary: pd.DataFrame, out: Path) -> None:
    df = pd.merge(model_summary, latency_summary, on="model_name", how="inner")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(df["avg_latency_ms"], df["avg_overall"], s=70, color="#6d597a")
    for _, r in df.iterrows():
        ax.annotate(r["model_name"], (r["avg_latency_ms"], r["avg_overall"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Latency vs Overall Quality (Model Means)")
    ax.set_xlabel("Average latency (ms)")
    ax.set_ylabel("Average overall quality")
    ax.set_ylim(0, 1)
    _save_fig(out)


def plot_quality_boxplot(scores: pd.DataFrame, out: Path) -> None:
    grouped = [g["overall_quality"].values for _, g in scores.groupby("model_name", sort=False)]
    labels = [k for k, _ in scores.groupby("model_name", sort=False)]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.boxplot(grouped, tick_labels=labels, showfliers=False)
    ax.set_title("Overall Quality Distribution by Model")
    ax.set_ylabel("Overall quality")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    _save_fig(out)


def plot_metric_correlation(scores: pd.DataFrame, out: Path) -> None:
    corr = scores[SCORE_COLS].corr().values
    if corr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Metric Correlation Matrix")
    ax.set_xticks(np.arange(len(SCORE_COLS)))
    ax.set_xticklabels(SCORE_COLS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(SCORE_COLS)))
    ax.set_yticklabels(SCORE_COLS, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save_fig(out)


def _required_outputs(run_dir: Path) -> list[Path]:
    charts_dir = run_dir / "insights" / "charts"
    data_dir = run_dir / "insights" / "data"
    return [
        charts_dir / "01_model_overall_quality_bar.png",
        charts_dir / "02_metric_heatmap_by_model.png",
        charts_dir / "03_language_overall_quality.png",
        charts_dir / "04_question_overall_quality.png",
        charts_dir / "05_latency_vs_quality_scatter.png",
        charts_dir / "06_overall_quality_boxplot_by_model.png",
        charts_dir / "07_metric_correlation_heatmap.png",
        data_dir / "model_summary.csv",
        data_dir / "language_summary.csv",
        data_dir / "question_summary.csv",
        data_dir / "latency_summary.csv",
    ]


def _is_complete(run_dir: Path) -> bool:
    return all(p.exists() for p in _required_outputs(run_dir))


def _discover_run_dirs(repo_root: Path) -> Iterable[Path]:
    runs_root = repo_root / "results" / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.glob("*/*") if p.is_dir()])


def _run_paths(run_dir: Path) -> dict:
    return {
        "run_dir": run_dir,
        "scores_db": run_dir / "scores" / "evaluation_scores_euf_context.db",
        "results_db": run_dir / "raw" / "evaluation_results_euf_context.db",
        "charts_dir": run_dir / "insights" / "charts",
        "data_dir": run_dir / "insights" / "data",
    }


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    if not force and _is_complete(run_dir):
        return False, f"skip (already complete): {run_dir}"

    rp = _run_paths(run_dir)
    missing_inputs = [p for p in [rp["scores_db"], rp["results_db"]] if not p.exists()]
    if missing_inputs:
        missing_text = ", ".join(str(p) for p in missing_inputs)
        return False, f"skip (missing input): {run_dir} :: {missing_text}"

    rp["charts_dir"].mkdir(parents=True, exist_ok=True)
    rp["data_dir"].mkdir(parents=True, exist_ok=True)

    try:
        scores = _read_table(rp["scores_db"], "scores")
        results = _read_table(rp["results_db"], "evaluations")
    except Exception as e:
        return False, f"skip (read error): {run_dir} :: {e}"

    for col in SCORE_COLS:
        if col in scores.columns:
            scores[col] = pd.to_numeric(scores[col], errors="coerce")
    if "latency_ms" in results.columns:
        results["latency_ms"] = pd.to_numeric(results["latency_ms"], errors="coerce")

    if scores.empty or results.empty:
        return False, f"skip (empty data): {run_dir}"

    scores["base_qid"] = scores["question_id"].map(_base_question_id)

    model_summary, language_summary, question_summary, latency_summary = build_summaries(scores, results)

    model_summary.to_csv(rp["data_dir"] / "model_summary.csv", index=False)
    language_summary.to_csv(rp["data_dir"] / "language_summary.csv", index=False)
    question_summary.to_csv(rp["data_dir"] / "question_summary.csv", index=False)
    latency_summary.to_csv(rp["data_dir"] / "latency_summary.csv", index=False)

    try:
        plot_model_overall_bar(model_summary, rp["charts_dir"] / "01_model_overall_quality_bar.png")
        plot_metric_heatmap(scores, rp["charts_dir"] / "02_metric_heatmap_by_model.png")
        plot_language_overall(language_summary, rp["charts_dir"] / "03_language_overall_quality.png")
        plot_question_overall(question_summary, rp["charts_dir"] / "04_question_overall_quality.png")
        plot_latency_vs_quality(model_summary, latency_summary, rp["charts_dir"] / "05_latency_vs_quality_scatter.png")
        plot_quality_boxplot(scores, rp["charts_dir"] / "06_overall_quality_boxplot_by_model.png")
        plot_metric_correlation(scores, rp["charts_dir"] / "07_metric_correlation_heatmap.png")
    except Exception as e:
        return False, f"skip (plot error): {run_dir} :: {e}"

    return True, f"generated: {run_dir}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate context charts/data for one run or all runs."
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
        help="Regenerate even when output artifacts already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        changed, msg = _process_run(run_dir, force=args.force)
        print(msg)
        if changed:
            rp = _run_paths(run_dir)
            print(f"Charts written to: {rp['charts_dir']}")
            print(f"Summary CSVs written to: {rp['data_dir']}")
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
