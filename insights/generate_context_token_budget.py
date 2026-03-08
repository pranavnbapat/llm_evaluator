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
from transformers import AutoTokenizer

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
        out_dir / "token_budget_question_profile.csv",
        out_dir / "token_budget_response_details.csv",
        out_dir / "token_budget_model_summary.csv",
        out_dir / "token_budget_language_question_summary.csv",
    ]


def _is_complete(out_dir: Path) -> bool:
    return all(p.exists() for p in _required_outputs(out_dir))


def _est_token_single(text: str) -> int:
    return int(round(len(text) / 4.0))


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


def _model_cfg_map_from_config(config_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not config_path.exists():
        return out
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        for _, cfg in (data.get("models") or {}).items():
            name = str(cfg.get("name", "")).strip()
            if not name:
                continue
            mlen = cfg.get("max_model_len")
            out[name.lower()] = {
                "max_model_len": mlen if isinstance(mlen, int) else DEFAULT_MAX_MODEL_LEN,
                "max_model_len_source": "config" if isinstance(mlen, int) else "assumed_from_config_default",
                "repo": str(cfg.get("repo", "")).strip(),
                "local_path": str(cfg.get("local_path", "")).strip(),
                "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
            }
    except Exception:
        pass
    return out


def _load_tokenizer_for_model(model_key: str, cfg_map: dict[str, dict], cache: dict[str, object]):
    if model_key in cache:
        return cache[model_key]

    cfg = cfg_map.get(model_key) or {}
    local_path = str(cfg.get("local_path") or "").strip()
    repo = str(cfg.get("repo") or "").strip()
    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    tok = None
    if local_path and Path(local_path).exists():
        try:
            tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        except Exception:
            tok = None
    if tok is None and repo:
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code)
        except Exception:
            tok = None

    cache[model_key] = tok
    return tok


def _count_tokens_exact(tokenizer, text: str) -> int:
    if tokenizer is None:
        return -1
    txt = "" if text is None else str(text)
    return len(tokenizer.encode(txt, add_special_tokens=False))


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    rp = _run_paths(run_dir)
    if not rp["results_db"].exists():
        return False, f"skip (missing input): {run_dir} :: {rp['results_db']}"

    out_dir = rp["out_dir"]
    if not force and _is_complete(out_dir):
        return False, f"skip (already complete): {run_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical question/profile metadata (language-agnostic planning view).
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
            }
        )
    qdf = pd.DataFrame(q_rows)

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
        .sort_values("base_question")
    )
    q_profile.to_csv(out_dir / "token_budget_question_profile.csv", index=False)

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

    cfg_map = _model_cfg_map_from_config(ROOT / "gpu_runtime" / "config.yaml")
    tokenizer_cache: dict[str, object] = {}

    m = res.merge(qdf[["question_id", "base_question"]], on="question_id", how="left")

    input_tokens = []
    response_tokens = []
    total_tokens = []
    max_len_vals = []
    max_len_src = []
    token_sources = []
    remaining_output = []
    effective_caps = []
    exceeds = []

    for _, row in m.iterrows():
        response = row.get("response") or ""
        question_text = row.get("question_text") or ""
        language = row.get("language") or ""
        context_text = row.get("context") or ""
        prompt_text = _build_prompt(str(question_text), str(language), str(context_text))

        model_key = str(row.get("model_name") or "").strip().lower()
        tok = _load_tokenizer_for_model(model_key, cfg_map, tokenizer_cache)

        in_tok = _count_tokens_exact(tok, prompt_text)
        out_tok = _count_tokens_exact(tok, response)
        if in_tok >= 0 and out_tok >= 0:
            token_sources.append("tokenizer_exact")
        else:
            token_sources.append("char_estimated")
            in_tok = _est_token_single(prompt_text)
            out_tok = _est_token_single(str(response))

        model_cfg = cfg_map.get(model_key)
        if model_cfg:
            max_len = int(model_cfg.get("max_model_len", DEFAULT_MAX_MODEL_LEN))
            src = str(model_cfg.get("max_model_len_source", "config"))
        else:
            max_len, src = DEFAULT_MAX_MODEL_LEN, "assumed_from_config_default"

        tot = int(in_tok) + int(out_tok)
        rem = int(max_len) - int(in_tok)
        cap = max(0, min(rem, 2048))

        input_tokens.append(int(in_tok))
        response_tokens.append(int(out_tok))
        total_tokens.append(int(tot))
        max_len_vals.append(int(max_len))
        max_len_src.append(src)
        remaining_output.append(int(rem))
        effective_caps.append(int(cap))
        exceeds.append(bool(tot > int(max_len)))

    m["input_tokens"] = input_tokens
    m["response_tokens"] = response_tokens
    m["total_tokens"] = total_tokens
    m["max_model_len"] = max_len_vals
    m["max_model_len_source"] = max_len_src
    m["token_count_source"] = token_sources
    m["remaining_output_tokens"] = remaining_output
    m["effective_output_cap"] = effective_caps
    m["would_exceed_max_len"] = exceeds

    detail_cols = [
        "id",
        "model_name",
        "language",
        "question_id",
        "base_question",
        "run_number",
        "input_tokens",
        "response_tokens",
        "total_tokens",
        "max_model_len",
        "max_model_len_source",
        "token_count_source",
        "remaining_output_tokens",
        "effective_output_cap",
        "would_exceed_max_len",
    ]
    m[detail_cols].to_csv(out_dir / "token_budget_response_details.csv", index=False)

    model_summary = (
        m.groupby("model_name", as_index=False)
        .agg(
            n=("id", "count"),
            max_model_len=("max_model_len", "max"),
            max_model_len_source=("max_model_len_source", "first"),
            input_tokens_mean=("input_tokens", "mean"),
            response_tokens_mean=("response_tokens", "mean"),
            response_tokens_p90=("response_tokens", lambda x: x.quantile(0.90)),
            response_tokens_max=("response_tokens", "max"),
            total_tokens_mean=("total_tokens", "mean"),
            total_tokens_max=("total_tokens", "max"),
            remaining_output_tokens_min=("remaining_output_tokens", "min"),
            remaining_output_tokens_avg=("remaining_output_tokens", "mean"),
            remaining_output_tokens_max=("remaining_output_tokens", "max"),
            effective_output_cap_min=("effective_output_cap", "min"),
            effective_output_cap_max=("effective_output_cap", "max"),
            exceed_rate=("would_exceed_max_len", "mean"),
        )
        .sort_values("model_name")
    )
    model_summary.to_csv(out_dir / "token_budget_model_summary.csv", index=False)

    lq_summary = (
        m.groupby(["language", "base_question"], as_index=False)
        .agg(
            input_tokens_mean=("input_tokens", "mean"),
            response_tokens_mean=("response_tokens", "mean"),
            response_tokens_p90=("response_tokens", lambda x: x.quantile(0.90)),
            response_tokens_max=("response_tokens", "max"),
            total_tokens_mean=("total_tokens", "mean"),
            total_tokens_max=("total_tokens", "max"),
        )
        .sort_values(["language", "base_question"])
    )
    lq_summary.to_csv(out_dir / "token_budget_language_question_summary.csv", index=False)

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
