#!/usr/bin/env python3
"""
Generate GPU efficiency report for context evaluation.

Single-run mode:
  python insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir <run_dir>

Bulk mode (default):
  python insights/gpu_efficiency/generate_gpu_efficiency_report.py
This scans results/runs/*/* and generates only missing GPU-efficiency artifacts.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def _read_sql(db: Path, q: str) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(q, con)


def _norm_model_name(v: str) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data_"
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def load_gpu_metrics(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {log_path}")
    df = pd.read_csv(log_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in [
        "util_gpu_pct",
        "util_mem_pct",
        "mem_total_mb",
        "mem_used_mb",
        "mem_free_mb",
        "temp_c",
        "cpu_util_pct",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "model_name" in df.columns:
        df["model_name_norm"] = df["model_name"].map(_norm_model_name)
    else:
        df["model_name_norm"] = ""
    return df


def build_tables(gpu: pd.DataFrame, results_db: Path, scores_db: Path) -> Dict[str, pd.DataFrame]:
    phase_counts = gpu["phase"].value_counts(dropna=False).rename_axis("phase").reset_index(name="samples")
    phase_counts["pct"] = (phase_counts["samples"] / phase_counts["samples"].sum() * 100).round(2)

    model_gpu = (
        gpu.dropna(subset=["model_name_norm"])
        .groupby(["model_name_norm"], as_index=False)
        .agg(
            samples=("timestamp", "count"),
            start_ts=("timestamp", "min"),
            end_ts=("timestamp", "max"),
            gpu_util_mean=("util_gpu_pct", "mean"),
            gpu_util_p10=("util_gpu_pct", lambda x: np.nanquantile(x, 0.1)),
            gpu_util_p90=("util_gpu_pct", lambda x: np.nanquantile(x, 0.9)),
            mem_used_p10_mb=("mem_used_mb", lambda x: np.nanquantile(x, 0.1)),
            mem_used_p90_mb=("mem_used_mb", lambda x: np.nanquantile(x, 0.9)),
            mem_used_mean_mb=("mem_used_mb", "mean"),
            mem_used_max_mb=("mem_used_mb", "max"),
            temp_max_c=("temp_c", "max"),
        )
    )
    model_gpu["duration_min"] = ((model_gpu["end_ts"] - model_gpu["start_ts"]).dt.total_seconds() / 60.0).round(2)
    model_gpu = model_gpu.sort_values("duration_min", ascending=False)

    results = _read_sql(
        results_db,
        """
        SELECT model_name, COUNT(*) AS n, AVG(latency_ms) AS avg_latency_ms,
               MAX(latency_ms) AS max_latency_ms
        FROM evaluations
        WHERE response IS NOT NULL
        GROUP BY model_name
        """,
    )
    results["model_name_norm"] = results["model_name"].map(_norm_model_name)

    scores = _read_sql(
        scores_db,
        """
        SELECT model_name, COUNT(*) AS n, AVG(overall_quality) AS avg_overall_quality
        FROM scores
        GROUP BY model_name
        """,
    )
    scores["model_name_norm"] = scores["model_name"].map(_norm_model_name)

    merged = model_gpu.merge(results[["model_name_norm", "model_name", "avg_latency_ms", "max_latency_ms"]], on="model_name_norm", how="left")
    merged = merged.merge(scores[["model_name_norm", "avg_overall_quality"]], on="model_name_norm", how="left")
    merged["display_model"] = merged["model_name"].fillna(merged["model_name_norm"])
    merged = merged.sort_values("gpu_util_mean", ascending=False)

    return {
        "phase_counts": phase_counts,
        "model_gpu": model_gpu,
        "model_efficiency": merged,
        "results_latency": results,
        "scores_quality": scores,
    }


def make_charts(t: Dict[str, pd.DataFrame], charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Phase share
    p = t["phase_counts"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.pie(p["samples"], labels=p["phase"], autopct="%1.1f%%", startangle=90)
    ax.set_title("GPU Metrics Phase Share")
    _save_fig(charts_dir / "01_phase_share_pie.png")

    # 2) Mean GPU util by model
    m = t["model_efficiency"].sort_values("gpu_util_mean", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(m["display_model"], m["gpu_util_mean"], color="#2a9d8f")
    ax.set_title("Mean GPU Utilization by Model")
    ax.set_ylabel("GPU Util (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    _save_fig(charts_dir / "02_gpu_util_mean_by_model.png")

    # 3) Max GPU memory used by model
    mm = t["model_efficiency"].sort_values("mem_used_max_mb", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(mm["display_model"], mm["mem_used_max_mb"] / 1024, color="#457b9d")
    ax.set_title("Max GPU Memory Used by Model")
    ax.set_ylabel("Memory (GB)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    _save_fig(charts_dir / "03_gpu_mem_max_by_model.png")

    # 4) Avg latency by model
    l = t["model_efficiency"].dropna(subset=["avg_latency_ms"]).sort_values("avg_latency_ms", ascending=True).head(12)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(l["display_model"], l["avg_latency_ms"], color="#e76f51")
    ax.set_title("Average Response Latency by Model")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    _save_fig(charts_dir / "04_latency_by_model.png")

    # 5) Latency vs overall quality
    s = t["model_efficiency"].dropna(subset=["avg_latency_ms", "avg_overall_quality"]).copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(s["avg_latency_ms"], s["avg_overall_quality"], s=70, color="#6d597a")
    for _, r in s.iterrows():
        ax.annotate(str(r["display_model"]), (r["avg_latency_ms"], r["avg_overall_quality"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("Latency vs Overall Quality")
    ax.set_xlabel("Average latency (ms)")
    ax.set_ylabel("Average overall quality")
    ax.set_ylim(0, 1)
    _save_fig(charts_dir / "05_latency_vs_quality_scatter.png")


def write_outputs(
    gpu: pd.DataFrame,
    t: Dict[str, pd.DataFrame],
    out_md: Path,
    data_dir: Path,
    source_log_label: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    for k, df in t.items():
        df.to_csv(data_dir / f"{k}.csv", index=False)

    start = gpu["timestamp"].min()
    end = gpu["timestamp"].max()
    duration_h = (end - start).total_seconds() / 3600.0 if pd.notna(start) and pd.notna(end) else np.nan

    # headline stats
    util_mean = gpu["util_gpu_pct"].mean()
    util_p10 = gpu["util_gpu_pct"].quantile(0.1)
    util_p90 = gpu["util_gpu_pct"].quantile(0.9)
    mem_mean_gb = gpu["mem_used_mb"].mean() / 1024.0
    mem_p10_gb = gpu["mem_used_mb"].quantile(0.1) / 1024.0
    mem_p90_gb = gpu["mem_used_mb"].quantile(0.9) / 1024.0
    mem_max_gb = gpu["mem_used_mb"].max() / 1024.0
    temp_max = gpu["temp_c"].max()

    model_eff = t["model_efficiency"].copy()
    best_quality = model_eff.dropna(subset=["avg_overall_quality"]).sort_values("avg_overall_quality", ascending=False).head(1)
    fastest = model_eff.dropna(subset=["avg_latency_ms"]).sort_values("avg_latency_ms", ascending=True).head(1)
    most_mem = model_eff.sort_values("mem_used_max_mb", ascending=False).head(1)
    least_mem = model_eff.sort_values("mem_used_max_mb", ascending=True).head(1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# GPU Efficiency Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Source log: `{source_log_label}`")
    lines.append(f"- Samples: **{len(gpu)}**")
    lines.append(f"- Time range: **{start} → {end}**")
    lines.append(f"- Duration: **{duration_h:.2f} hours**")
    lines.append("")
    lines.append("## Core GPU Statistics")
    lines.append("")
    lines.append(f"- Mean GPU utilization: **{util_mean:.2f}%**")
    lines.append(f"- P10 GPU utilization: **{util_p10:.2f}%**")
    lines.append(f"- P90 GPU utilization: **{util_p90:.2f}%**")
    lines.append(f"- Mean GPU memory used: **{mem_mean_gb:.2f} GB**")
    lines.append(f"- P10 GPU memory used: **{mem_p10_gb:.2f} GB**")
    lines.append(f"- P90 GPU memory used: **{mem_p90_gb:.2f} GB**")
    lines.append(f"- Max GPU memory used: **{mem_max_gb:.2f} GB**")
    lines.append(f"- Max GPU temperature: **{temp_max:.1f}°C**")
    lines.append("")
    lines.append("## Metric Glossary (Simple Terms)")
    lines.append("")
    lines.append("- **`gpu_util_mean`**: average GPU usage over time for a model.")
    lines.append("  Example: `70%` means the GPU was, on average, 70% busy while that model was active.")
    lines.append("- **`gpu_util_p90`**: 90th percentile GPU usage.")
    lines.append("  Simple meaning: for 90% of samples, usage was at or below this value; only 10% were higher.")
    lines.append("- **`gpu_util_p10`**: 10th percentile GPU usage.")
    lines.append("  Simple meaning: 10% of samples were at or below this value; this shows low-end utilization periods.")
    lines.append("- **`mem_used_max_gb`**: highest GPU memory observed for a model.")
    lines.append("  Use this for capacity planning and OOM risk checks.")
    lines.append("- **`mem_used_mean_gb`**: average GPU memory used for a model.")
    lines.append("  Helps compare typical memory footprint across models.")
    lines.append("- **`mem_used_p90_gb`**: high-memory operating level for a model (not just one-off peak).")
    lines.append("- **`temp_max_c`**: highest GPU temperature observed.")
    lines.append("- **`avg_latency_ms`**: average end-to-end response latency per model from results DB.")
    lines.append("- **`avg_overall_quality`**: average final quality score from scores DB (0 to 1).")
    lines.append("")
    lines.append("### Why Percentiles (P10/P90) Matter")
    lines.append("")
    lines.append("- `mean` can hide spikes and dips.")
    lines.append("- `P90` shows whether high utilization/memory is sustained or only occasional.")
    lines.append("- `P10` shows lower-tail behavior (idle gaps, stalls, loading transitions).")
    lines.append("- Quick rule: if `P90` is very high and `P10` is very low, workload is bursty.")
    lines.append("")
    lines.append("### Example Interpretation")
    lines.append("")
    lines.append(
        f"- In this run, global `gpu_util_p90` is **{util_p90:.2f}%**, so the GPU was near saturation for most active samples."
    )
    lines.append(
        f"- Global `mem_used_max_gb` is **{mem_max_gb:.2f} GB** while `mem_used_mean_gb` is **{mem_mean_gb:.2f} GB`."
    )
    lines.append("  This indicates high memory residency during evaluation phases with lower memory during loading/idle.")
    lines.append("")

    if not best_quality.empty:
        r = best_quality.iloc[0]
        lines.append("## Performance Highlights")
        lines.append("")
        lines.append(f"- Best model overall quality: **{r['display_model']}** ({r['avg_overall_quality']:.3f})")
    if not fastest.empty:
        r = fastest.iloc[0]
        lines.append(f"- Fastest responses (avg latency): **{r['display_model']}** ({r['avg_latency_ms']:.1f} ms)")
    if not most_mem.empty:
        r = most_mem.iloc[0]
        lines.append(f"- Most GPU memory consumed (max): **{r['display_model']}** ({r['mem_used_max_mb']/1024:.2f} GB)")
    if not least_mem.empty:
        r = least_mem.iloc[0]
        lines.append(f"- Minimum GPU memory consumed (max): **{r['display_model']}** ({r['mem_used_max_mb']/1024:.2f} GB)")
    lines.append("")

    lines.append("## Phase Breakdown")
    lines.append("")
    p = t["phase_counts"].copy()
    p["pct"] = p["pct"].map(lambda x: f"{x:.2f}")
    lines.append(_md_table(p[["phase", "samples", "pct"]]))
    lines.append("")

    lines.append("## Model-Level GPU Efficiency")
    lines.append("")
    m = t["model_efficiency"].copy()
    m["gpu_util_p10"] = m["gpu_util_p10"].map(lambda x: f"{x:.2f}")
    m["gpu_util_mean"] = m["gpu_util_mean"].map(lambda x: f"{x:.2f}")
    m["gpu_util_p90"] = m["gpu_util_p90"].map(lambda x: f"{x:.2f}")
    m["mem_used_p10_gb"] = (m["mem_used_p10_mb"] / 1024.0).map(lambda x: f"{x:.2f}")
    m["mem_used_mean_gb"] = (m["mem_used_mean_mb"] / 1024.0).map(lambda x: f"{x:.2f}")
    m["mem_used_p90_gb"] = (m["mem_used_p90_mb"] / 1024.0).map(lambda x: f"{x:.2f}")
    m["mem_used_max_gb"] = (m["mem_used_max_mb"] / 1024.0).map(lambda x: f"{x:.2f}")
    m["avg_latency_ms"] = m["avg_latency_ms"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "NA")
    m["avg_overall_quality"] = m["avg_overall_quality"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
    lines.append(
        _md_table(
            m[
                [
                    "display_model",
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
                ]
            ].rename(columns={"display_model": "model"})
        )
    )
    lines.append("")

    lines.append("## Generated Artifacts")
    lines.append("")
    lines.append("- Charts: `insights/gpu_efficiency/charts/`")
    lines.append("- Data tables: `insights/gpu_efficiency/data/`")
    lines.append("- This report: `insights/gpu_efficiency/GPU_EFFICIENCY_REPORT.md`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def _required_outputs(out_dir: Path) -> list[Path]:
    charts_dir = out_dir / "charts"
    data_dir = out_dir / "data"
    return [
        out_dir / "GPU_EFFICIENCY_REPORT.md",
        charts_dir / "01_phase_share_pie.png",
        charts_dir / "02_gpu_util_mean_by_model.png",
        charts_dir / "03_gpu_mem_max_by_model.png",
        charts_dir / "04_latency_by_model.png",
        charts_dir / "05_latency_vs_quality_scatter.png",
        data_dir / "phase_counts.csv",
        data_dir / "model_gpu.csv",
        data_dir / "model_efficiency.csv",
        data_dir / "results_latency.csv",
        data_dir / "scores_quality.csv",
    ]


def _is_complete(out_dir: Path) -> bool:
    return all(p.exists() for p in _required_outputs(out_dir))


def _discover_run_dirs(repo_root: Path) -> Iterable[Path]:
    runs_root = repo_root / "results" / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.glob("*/*") if p.is_dir()])


def _run_paths(run_dir: Path) -> dict:
    out_dir = run_dir / "insights" / "gpu_efficiency"
    return {
        "run_dir": run_dir,
        "log_path": run_dir / "logs" / "gpu_metrics.csv",
        "results_db": run_dir / "raw" / "evaluation_results_euf_context.db",
        "scores_db": run_dir / "scores" / "evaluation_scores_euf_context.db",
        "out_dir": out_dir,
        "charts_dir": out_dir / "charts",
        "data_dir": out_dir / "data",
        "out_md": out_dir / "GPU_EFFICIENCY_REPORT.md",
    }


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    rp = _run_paths(run_dir)
    out_dir = rp["out_dir"]
    if not force and _is_complete(out_dir):
        return False, f"skip (already complete): {run_dir}"

    missing_inputs = [p for p in [rp["log_path"], rp["results_db"], rp["scores_db"]] if not p.exists()]
    if missing_inputs:
        missing_text = ", ".join(str(p) for p in missing_inputs)
        return False, f"skip (missing input): {run_dir} :: {missing_text}"

    rp["out_dir"].mkdir(parents=True, exist_ok=True)
    rp["charts_dir"].mkdir(parents=True, exist_ok=True)
    rp["data_dir"].mkdir(parents=True, exist_ok=True)

    gpu = load_gpu_metrics(rp["log_path"])
    tables = build_tables(gpu, rp["results_db"], rp["scores_db"])
    make_charts(tables, rp["charts_dir"])
    write_outputs(
        gpu,
        tables,
        rp["out_md"],
        rp["data_dir"],
        "logs/gpu_metrics.csv",
    )
    return True, f"generated: {run_dir}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPU-efficiency artifacts for one run or all runs."
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
            print(f"Wrote report: {rp['out_md']}")
            print(f"Wrote charts: {rp['charts_dir']}")
            print(f"Wrote data: {rp['data_dir']}")
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
