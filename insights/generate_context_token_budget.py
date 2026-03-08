#!/usr/bin/env python3
"""
Generate token-budget CSV artifacts for context-evaluation runs.

Single-run mode:
  python insights/generate_context_token_budget.py --run-dir <run_dir>

Bulk mode (default):
  python insights/generate_context_token_budget.py
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translations.eu_24_languages_euf_context import get_all_questions_with_context

DEFAULT_MAX_MODEL_LEN = 16384


def _discover_run_dirs(repo_root: Path) -> Iterable[Path]:
    runs_root = repo_root / "results" / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.glob("*/*") if p.is_dir()])


def _run_paths(run_dir: Path) -> dict:
    return {
        "run_dir": run_dir,
        "results_db": run_dir / "raw" / "evaluation_results_euf_context.db",
        "out_dir": run_dir / "insights" / "data",
    }


def _required_outputs(out_dir: Path) -> list[Path]:
    return [
        out_dir / "token_budget_prompt_details_estimated.csv",
        out_dir / "token_budget_question_profile_estimated.csv",
        out_dir / "token_budget_language_question_estimated.csv",
        out_dir / "token_budget_model_language_question_estimated.csv",
        out_dir / "token_budget_model_output_budget_estimated.csv",
        out_dir / "token_budget_response_details_estimated_range.csv",
        out_dir / "token_budget_response_model_summary_estimated_range.csv",
        out_dir / "token_budget_response_language_question_summary_estimated_range.csv",
    ]


def _is_complete(out_dir: Path) -> bool:
    return all(p.exists() for p in _required_outputs(out_dir))


def _est_token_single(text: str) -> int:
    return int(round(len(text) / 4.0))


def _est_token_range(text: str) -> tuple[int, int, int]:
    n = len(text)
    t_min = int(math.ceil(n / 5.0))
    t_max = int(math.ceil(n / 4.0))
    t_mid = int(round((t_min + t_max) / 2.0))
    return t_min, t_max, t_mid


def _build_prompt(question_text: str, language: str, context_str: str) -> str:
    return f"""You are an expert agriculture advisor. A farmer has asked you a question. Use the provided search results to give a helpful, accurate response.

SEARCH RESULTS (in English):
{context_str}

FARMER'S QUESTION (in {language}):
{question_text}

IMPORTANT INSTRUCTIONS:
1. Answer in the SAME LANGUAGE as the farmer's question ({language})
2. Provide a COMPREHENSIVE but CONCISE answer (2-4 paragraphs)
3. Use SPECIFIC details from the search results
4. Give PRACTICAL, actionable advice that farmers can implement
5. If the search results don't fully answer the question, provide your best expert knowledge

Your response:"""


def _format_context(context: list) -> str:
    parts = []
    for i, entry in enumerate(context or [], 1):
        title = entry.get("title", "")
        description = entry.get("description", "")
        if title and description:
            parts.append(f"[{i}] {title}: {description[:300]}...")
        elif title:
            parts.append(f"[{i}] {title}")
    return "\n\n".join(parts)


def _model_len_map_from_config(config_path: Path) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    if not config_path.exists():
        return out
    try:
        import yaml
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for _, cfg in (data.get("models") or {}).items():
            name = str(cfg.get("name", "")).strip()
            mlen = cfg.get("max_model_len")
            if name and isinstance(mlen, int):
                out[name.lower()] = (mlen, "config")
    except Exception:
        pass
    return out


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    rp = _run_paths(run_dir)
    if not rp["results_db"].exists():
        return False, f"skip (missing input): {run_dir} :: {rp['results_db']}"

    out_dir = rp["out_dir"]
    if not force and _is_complete(out_dir):
        return False, f"skip (already complete): {run_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prompt/question context estimates from canonical question source.
    q_rows = []
    for q in get_all_questions_with_context():
        qid = q["question_id"]
        lang = q["language"]
        base_q = qid.split("_", 1)[0]
        qtxt = q["question"]
        cstr = _format_context(q.get("context", []))
        prompt = _build_prompt(qtxt, lang, cstr)
        q_rows.append(
            {
                "question_id": qid,
                "base_question": base_q,
                "language": lang,
                "question_tokens_est": _est_token_single(qtxt),
                "context_tokens_est": _est_token_single(cstr),
                "prompt_tokens_est": _est_token_single(prompt),
                "question_chars": len(qtxt),
                "context_chars": len(cstr),
                "prompt_chars": len(prompt),
            }
        )
    qdf = pd.DataFrame(q_rows)

    prompt_details = qdf[[
        "question_id", "base_question", "language", "question_tokens_est", "context_tokens_est", "prompt_tokens_est"
    ]].sort_values(["base_question", "language"])
    prompt_details.to_csv(out_dir / "token_budget_prompt_details_estimated.csv", index=False)

    q_profile = (
        qdf.groupby("base_question", as_index=False)
        .agg(
            context_tokens_est=("context_tokens_est", "first"),
            question_tokens_est_min=("question_tokens_est", "min"),
            question_tokens_est_avg=("question_tokens_est", "mean"),
            question_tokens_est_max=("question_tokens_est", "max"),
            prompt_tokens_est_min=("prompt_tokens_est", "min"),
            prompt_tokens_est_avg=("prompt_tokens_est", "mean"),
            prompt_tokens_est_max=("prompt_tokens_est", "max"),
        )
    )
    q_profile.to_csv(out_dir / "token_budget_question_profile_estimated.csv", index=False)

    lq = qdf[["language", "base_question", "question_tokens_est", "context_tokens_est", "prompt_tokens_est"]].sort_values(["language", "base_question"])
    lq.to_csv(out_dir / "token_budget_language_question_estimated.csv", index=False)

    # Response-level estimates from results DB.
    with sqlite3.connect(rp["results_db"]) as con:
        res = pd.read_sql_query(
            """
            SELECT id, model_name, language, question_id, run_number, question_text, context, response
            FROM evaluations
            WHERE response IS NOT NULL
            ORDER BY id
            """,
            con,
        )

    if res.empty:
        for p in _required_outputs(out_dir):
            if not p.exists():
                pd.DataFrame().to_csv(p, index=False)
        return True, f"generated (empty data): {run_dir}"

    config_map = _model_len_map_from_config(ROOT / "gpu_runtime" / "config.yaml")

    m = res.merge(
        qdf[["question_id", "base_question", "question_chars", "context_chars", "prompt_chars", "prompt_tokens_est"]],
        on="question_id",
        how="left",
    )

    # Token ranges for prompt/response/total.
    rmins = []
    rmaxs = []
    rmids = []
    tmins = []
    tmaxs = []
    tmids = []
    remain_min = []
    remain_max = []
    cap_min = []
    cap_max = []
    max_len_vals = []
    max_len_src = []

    for _, row in m.iterrows():
        response = row.get("response") or ""
        pchars = int(row.get("prompt_chars") or 0)
        rchars = len(response)

        in_min, in_max, in_mid = _est_token_range("x" * pchars)
        out_min, out_max, out_mid = _est_token_range(response)

        model_key = str(row.get("model_name") or "").strip().lower()
        if model_key in config_map:
            max_len, src = config_map[model_key]
        else:
            max_len, src = DEFAULT_MAX_MODEL_LEN, "assumed_from_config_default"

        total_min = in_min + out_min
        total_max = in_max + out_max
        total_mid = in_mid + out_mid

        rem_min = max_len - in_max
        rem_max = max_len - in_min
        eff_cap_min = max(0, min(rem_min, 2048))
        eff_cap_max = max(0, min(rem_max, 2048))

        rmins.append(out_min)
        rmaxs.append(out_max)
        rmids.append(out_mid)
        tmins.append(total_min)
        tmaxs.append(total_max)
        tmids.append(total_mid)
        remain_min.append(rem_min)
        remain_max.append(rem_max)
        cap_min.append(eff_cap_min)
        cap_max.append(eff_cap_max)
        max_len_vals.append(max_len)
        max_len_src.append(src)

    m["response_chars"] = m["response"].map(lambda x: len(x or ""))
    m["input_tokens_est_min"] = m["prompt_chars"].map(lambda n: int(math.ceil((n or 0) / 5.0)))
    m["input_tokens_est_max"] = m["prompt_chars"].map(lambda n: int(math.ceil((n or 0) / 4.0)))
    m["input_tokens_est_mid"] = ((m["input_tokens_est_min"] + m["input_tokens_est_max"]) / 2.0).round().astype(int)
    m["response_tokens_est_min"] = rmins
    m["response_tokens_est_max"] = rmaxs
    m["response_tokens_est_mid"] = rmids
    m["total_tokens_est_min"] = tmins
    m["total_tokens_est_max"] = tmaxs
    m["total_tokens_est_mid"] = tmids
    m["max_model_len"] = max_len_vals
    m["max_model_len_source"] = max_len_src
    m["remaining_output_tokens_est_min"] = remain_min
    m["remaining_output_tokens_est_max"] = remain_max
    m["effective_output_cap_est_min"] = cap_min
    m["effective_output_cap_est_max"] = cap_max
    m["would_exceed_max_len_est"] = m["total_tokens_est_mid"] > m["max_model_len"]

    resp_detail_cols = [
        "id", "model_name", "language", "question_id", "run_number",
        "question_chars", "context_chars", "prompt_chars", "response_chars",
        "input_tokens_est_min", "input_tokens_est_max", "input_tokens_est_mid",
        "response_tokens_est_min", "response_tokens_est_max", "response_tokens_est_mid",
        "total_tokens_est_min", "total_tokens_est_max", "total_tokens_est_mid",
        "max_model_len", "max_model_len_source",
        "remaining_output_tokens_est_min", "remaining_output_tokens_est_max",
        "effective_output_cap_est_min", "effective_output_cap_est_max",
        "would_exceed_max_len_est", "base_question",
    ]
    m[resp_detail_cols].to_csv(out_dir / "token_budget_response_details_estimated_range.csv", index=False)

    model_summary = (
        m.groupby("model_name", as_index=False)
        .agg(
            n=("id", "count"),
            max_model_len=("max_model_len", "max"),
            max_model_len_source=("max_model_len_source", "first"),
            input_tokens_est_mid_mean=("input_tokens_est_mid", "mean"),
            response_tokens_est_mid_mean=("response_tokens_est_mid", "mean"),
            response_tokens_est_mid_p90=("response_tokens_est_mid", lambda x: x.quantile(0.90)),
            response_tokens_est_mid_max=("response_tokens_est_mid", "max"),
            total_tokens_est_mid_mean=("total_tokens_est_mid", "mean"),
            total_tokens_est_mid_max=("total_tokens_est_mid", "max"),
            remaining_output_tokens_est_min=("remaining_output_tokens_est_min", "min"),
            remaining_output_tokens_est_max=("remaining_output_tokens_est_max", "max"),
        )
        .sort_values("model_name")
    )
    model_summary.to_csv(out_dir / "token_budget_response_model_summary_estimated_range.csv", index=False)

    lq_resp = (
        m.groupby(["language", "base_question"], as_index=False)
        .agg(
            input_tokens_est_mid=("input_tokens_est_mid", "mean"),
            response_tokens_est_mid_mean=("response_tokens_est_mid", "mean"),
            response_tokens_est_mid_p90=("response_tokens_est_mid", lambda x: x.quantile(0.90)),
            response_tokens_est_mid_max=("response_tokens_est_mid", "max"),
            total_tokens_est_mid_mean=("total_tokens_est_mid", "mean"),
            total_tokens_est_mid_max=("total_tokens_est_mid", "max"),
        )
        .sort_values(["language", "base_question"])
    )
    lq_resp.to_csv(out_dir / "token_budget_response_language_question_summary_estimated_range.csv", index=False)

    # Prompt-budget by model + question/language.
    mlq = m[["question_id", "language", "base_question", "prompt_tokens_est", "model_name", "max_model_len"]].copy()
    mlq["remaining_output_tokens_est"] = mlq["max_model_len"] - mlq["prompt_tokens_est"]
    mlq["effective_output_cap"] = mlq["remaining_output_tokens_est"].map(lambda x: max(0, min(int(x), 2048)))
    mlq = mlq.drop_duplicates(subset=["question_id", "model_name"]).sort_values(["model_name", "question_id"])
    mlq.to_csv(out_dir / "token_budget_model_language_question_estimated.csv", index=False)

    mo = (
        mlq.groupby("model_name", as_index=False)
        .agg(
            max_model_len=("max_model_len", "max"),
            remaining_output_tokens_est_min=("remaining_output_tokens_est", "min"),
            remaining_output_tokens_est_avg=("remaining_output_tokens_est", "mean"),
            remaining_output_tokens_est_max=("remaining_output_tokens_est", "max"),
            effective_output_cap_min=("effective_output_cap", "min"),
            effective_output_cap_max=("effective_output_cap", "max"),
        )
        .sort_values("model_name")
    )
    mo.to_csv(out_dir / "token_budget_model_output_budget_estimated.csv", index=False)

    return True, f"generated: {run_dir}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate token-budget CSV artifacts for one run or all runs.")
    parser.add_argument("--run-dir", type=Path, help="Specific run directory (e.g., results/runs/a40/<run_id>).")
    parser.add_argument("--all-runs", action="store_true", help="Process all runs under results/runs/*/* (default when --run-dir is omitted).")
    parser.add_argument("--force", action="store_true", help="Regenerate even when outputs already exist.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        changed, msg = _process_run(run_dir, force=args.force)
        print(msg)
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
