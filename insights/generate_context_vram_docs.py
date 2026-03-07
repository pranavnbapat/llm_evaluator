#!/usr/bin/env python3
"""
Generate VRAM/concurrency markdown docs and FAQs for context-evaluation runs.

Single-run mode:
  python insights/generate_context_vram_docs.py --run-dir <run_dir>

Bulk mode (default):
  python insights/generate_context_vram_docs.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _discover_run_dirs(repo_root: Path) -> Iterable[Path]:
    runs_root = repo_root / "results" / "runs"
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.glob("*/*") if p.is_dir()])


def _run_paths(run_dir: Path) -> dict:
    data_dir = run_dir / "insights" / "data"
    return {
        "run_dir": run_dir,
        "insights_dir": run_dir / "insights",
        "data_dir": data_dir,
        "model_summary": data_dir / "model_summary.csv",
        "language_summary": data_dir / "language_summary.csv",
        "question_summary": data_dir / "question_summary.csv",
        "latency_summary": data_dir / "latency_summary.csv",
        "token_model_summary": data_dir / "token_budget_response_model_summary_estimated_range.csv",
        "token_q_profile": data_dir / "token_budget_question_profile_estimated.csv",
        "token_lq_summary": data_dir / "token_budget_response_language_question_summary_estimated_range.csv",
        "token_response_detail": data_dir / "token_budget_response_details_estimated_range.csv",
        "token_model_output_budget": data_dir / "token_budget_model_output_budget_estimated.csv",
        "guide_md": run_dir / "insights" / "VRAM_CONTEXT_CONCURRENCY_GUIDE.md",
        "deep_dive_md": run_dir / "insights" / "VRAM_CONTEXT_CONCURRENCY_DEEP_DIVE.md",
        "faq_md": run_dir / "insights" / "FAQs.md",
    }


def _required_outputs(rp: dict) -> list[Path]:
    return [rp["guide_md"], rp["deep_dive_md"], rp["faq_md"]]


def _is_complete(rp: dict) -> bool:
    return all(p.exists() for p in _required_outputs(rp))


def _load_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_top(df: pd.DataFrame, sort_col: str, asc: bool = False):
    if df.empty or sort_col not in df.columns:
        return None
    return df.sort_values(sort_col, ascending=asc).iloc[0]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "_No data_"
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return "_No data_"
    d = df[keep].head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.3f}")
    lines = []
    lines.append("| " + " | ".join(d.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(d.columns)) + " |")
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in d.columns) + " |")
    return "\n".join(lines)


def _render_guide(ctx: dict) -> str:
    return f"""# VRAM, Context Window, and Concurrency Guide

Generated: {ctx['ts']}

This note explains what consumes GPU memory, what context window means in practice, and how to estimate parallel-user capacity.

## 1) What Actually Sits in VRAM

When an LLM is running, VRAM is mainly used by:

1. Model weights
- Fixed size once loaded.
- Rough FP16 intuition: `7B ~ 14 GB`, `30B ~ 60 GB`.
- Quantization reduces this.

2. KV cache (working memory)
- Grows with active tokens.
- Depends on sequence length, layers, hidden/KV dimensions, and precision.
- Main reason long context causes OOM.

3. Runtime buffers
- Non-zero overhead (scheduler, kernels, temporary tensors).

## 2) Context Window in Plain English

Context window = total tokens visible at once:

`system prompt + user input + retrieved RAG text + chat history + generated output`

So for an 8K model:
`input_tokens + output_tokens <= 8192`

## 3) Why Context Affects Memory

KV cache scales with token count, so:
- 1K tokens -> small cache
- 32K tokens -> large cache

Same model + same GPU can fail at longer context.

## 4) Multi-User Concurrency: Core Rule

Weights are shared across users, KV cache is not.

With `N` concurrent users:
`VRAM = model_weights + KV_1 + KV_2 + ... + KV_N + runtime_overhead`

## 5) This Run: Practical Profile

- Mean input tokens (estimated): **{ctx['input_mean']}**
- Mean response tokens (estimated): **{ctx['resp_mean']}**
- P90 response tokens (estimated): **{ctx['resp_p90']}**
- Max observed total tokens (estimated): **{ctx['total_max']}**

Top model by quality: **{ctx['best_model']}** ({ctx['best_score']})  
Fastest model by latency: **{ctx['fast_model']}** ({ctx['fast_latency']})

## 6) Model Ranking Snapshot

{ctx['model_table']}

## 7) Language Snapshot

{ctx['lang_table']}

## 8) Question-Family Snapshot

{ctx['question_table']}

## 9) A40 Planning Note

For A40 (48 GB), multi-user capacity depends primarily on per-request token depth.
Use conservative output caps and bounded retrieved context to keep KV cache stable.

## 10) One-Line Summary

Concurrency is not only about model size; it is mostly a KV-cache budget problem under real prompt+output lengths.

## 11) A40-Friendly Alternatives (This Run)

{ctx['model_table']}
"""


def _render_deep_dive(ctx: dict) -> str:
    return f"""# VRAM, Context Window, and Concurrency Deep Dive

Generated: {ctx['ts']}

This report explains VRAM, tokens, context windows, and concurrency using practical serving behavior. The focus is operational mechanics: what consumes memory, why OOM happens, and how to size systems reliably.

## 1) What actually sits in VRAM

When a model is loaded on a GPU, three major things consume VRAM.

### (A) Model weights
This is the model brain. Fixed size after load.

- A 7B model in FP16 is roughly 14 GB.
- A 30B model in FP16 is roughly 60 GB.
- Quantized models reduce this.

This part does not grow with prompt length.

### (B) KV cache (working memory)
This is where context lives while a request runs.

It grows with:
- number of active tokens in the request
- number of layers
- hidden/KV dimensions
- precision

This is why long context can trigger OOM even if the model itself fits.

### (C) Temporary runtime buffers
Non-zero memory for kernels, intermediate tensors, and serving overhead.

## 2) What context window means in plain terms

Context window is the total token budget visible at once:

`system prompt + user input + retrieved RAG context + chat history + generated output`

So for an 8K model:

`input tokens + output tokens <= 8192`

It is not input-only.

## 3) Tokens to text intuition

Very rough intuition:
- 1 token is about 3/4 of a word
- 100 tokens is about a short paragraph
- 1000 tokens is about a page

## 4) Why context directly affects GPU memory

KV cache stores attention state per token.

So:
- 1K tokens: small KV usage
- 32K tokens: very large KV usage

This is why the same model on the same GPU can run fine at short context and fail at long context.

## 5) Real RAG scenario end-to-end

User asks:

"What are the soil carbon benefits of crop rotation in Mediterranean climates?"

Approximate token accounting:

1. System prompt: ~120
2. User question: ~20
3. Retrieved context: 3 chunks x 500 = 1500
4. Formatting instructions: ~80

Total input = `120 + 20 + 1500 + 80 = 1720`

If output is ~600 tokens:

Total context usage = `1720 + 600 = 2320`

Fit check:
- 4K context model: OK
- 2K context model: likely truncation/failure

## 6) GPU selection logic

Step 1: pick GPU VRAM budget.

Example:
- A40: 48 GB VRAM

Step 2: check model weight fit.

Step 3: check remaining VRAM for KV cache.

This determines usable context and concurrency.

## 7) Why output length also matters

During generation, output tokens are appended to active context.

So if `max_new_tokens` is high, you must reserve memory for that growth. Otherwise generation can fail mid-response.

## 8) Core idea for stakeholders

GPU sizing is not only "can I load the model?"

It is:
"Can I load the model and keep enough working memory for the context depth my use case needs?"

## 9) One-sentence truth

Context window is not only a model property; in production it is a VRAM budget problem.

---

# Multi-user concurrency: what changes

When many users are active simultaneously, weights are shared, KV cache is not.

For `N` concurrent requests:

`VRAM = weights + KV_1 + KV_2 + ... + KV_N + runtime_overhead`

## Why long context reduces concurrency

Short requests -> small KV per user -> more users fit.  
Long RAG requests -> large KV per user -> fewer users fit.

## Throughput vs latency

- High concurrency mode: short prompts/outputs.
- Deep RAG mode: long prompts/outputs, fewer users.

## Continuous batching (vLLM)

Continuous batching improves throughput and utilization by mixing tokens from many requests into shared forward passes.
But it does not eliminate KV memory limits. KV still sets the hard concurrency cap.

---

# Token Budget Analysis (Current Context Evaluation)

This section computes token budgets from run artifacts and prompt/question datasets.

Approximation used:
- **1 token ~= 4 to 5 characters**
- For each text, this analysis keeps min/max/mid estimates.

Definitions:
- **Total input tokens** = full prompt tokens (instructions + context + question)
- **Response tokens** = generated response tokens
- **Total sequence tokens** = input + response
- **Remaining output tokens** = `max_model_len - input_tokens`

| Metric | Value |
| --- | ---: |
| Mean estimated input tokens | {ctx['input_mean']} |
| Mean estimated response tokens | {ctx['resp_mean']} |
| P90 estimated response tokens | {ctx['resp_p90']} |
| Max estimated total tokens | {ctx['total_max']} |

## A) Input/Output Token Profile by Base Question (from real responses)

{ctx['token_q_table_extended']}

## B) Per-Model Output Budget Using Config Max Context

{ctx['token_model_output_budget_table']}

## C) Per-Language x Per-Question (Input + Response + Total)

{ctx['token_lq_table']}

## D) Full Row-Level Token Table (All Responses)

A detailed per-response table is exported to CSV:
- `insights/data/token_budget_response_details_estimated_range.csv` ({ctx['token_row_count']} rows)

## E) CSV Exports

- `insights/data/token_budget_response_details_estimated_range.csv`
- `insights/data/token_budget_response_language_question_summary_estimated_range.csv`
- `insights/data/token_budget_response_model_summary_estimated_range.csv`
- `insights/data/token_budget_model_language_question_estimated.csv`
- `insights/data/token_budget_question_profile_estimated.csv`
- `insights/data/token_budget_model_output_budget_estimated.csv`
- `insights/data/token_budget_language_question_estimated.csv`
- `insights/data/token_budget_prompt_details_estimated.csv`

---

## Quality/Latency highlights

| Item | Value |
| --- | --- |
| Best model by overall quality | {ctx['best_model']} ({ctx['best_score']}) |
| Fastest model by mean latency | {ctx['fast_model']} ({ctx['fast_latency']}) |
| Top language | {ctx['top_lang']} ({ctx['top_lang_score']}) |
| Bottom language | {ctx['bottom_lang']} ({ctx['bottom_lang_score']}) |
| Top question family | {ctx['top_q']} ({ctx['top_q_score']}) |
| Lowest question family | {ctx['low_q']} ({ctx['low_q_score']}) |

## Why Concurrency Drops

For `N` concurrent requests:
`VRAM = weights + KV_1 + ... + KV_N + runtime_overhead`

KV cache is per-user and scales with active tokens.  
Longer prompts/outputs -> larger KV per request -> lower safe concurrency.

## Operational Interpretation

- If concurrency target rises, keep `max_tokens` and retrieved context bounded.
- If answer depth is prioritized, reduce concurrent sessions.
- Prefer queueing/backpressure over hard failures under burst load.

## Recommendation

Use generated token-budget CSVs in `insights/data/` as the canonical baseline for capacity planning and model comparison.
"""


def _render_faq(ctx: dict) -> str:
    return f"""# FAQs

Generated: {ctx['ts']}

## What did we evaluate?
Context-grounded responses across models, 24 EU languages, and 5 question families.

## Which model performed best overall?
{ctx['best_model']} with average overall quality {ctx['best_score']}.

## Which model was fastest?
{ctx['fast_model']} with average latency {ctx['fast_latency']}.

## What token lengths did we observe?
Mean input ~{ctx['input_mean']} tokens, mean output ~{ctx['resp_mean']} tokens, p90 output ~{ctx['resp_p90']}.

## Why does concurrency drop with long context?
Because KV cache grows with active tokens per request and is not shared across users.

## Is this enough for production sizing?
Use this as baseline only. Validate with a short load test using your exact retrieval chunking and output cap.
"""


def _process_run(run_dir: Path, force: bool = False) -> tuple[bool, str]:
    rp = _run_paths(run_dir)
    if not force and _is_complete(rp):
        return False, f"skip (already complete): {run_dir}"

    msum = _load_or_empty(rp["model_summary"])
    lsum = _load_or_empty(rp["language_summary"])
    qsum = _load_or_empty(rp["question_summary"])
    lat = _load_or_empty(rp["latency_summary"])
    tmodel = _load_or_empty(rp["token_model_summary"])
    tq = _load_or_empty(rp["token_q_profile"])
    tlq = _load_or_empty(rp["token_lq_summary"])
    trow = _load_or_empty(rp["token_response_detail"])
    tmodel_out = _load_or_empty(rp["token_model_output_budget"])

    # Minimal input requirement: at least one summary table.
    if msum.empty and tmodel.empty:
        return False, f"skip (missing input): {run_dir} :: insights/data summaries not found"

    best = _safe_top(msum, "avg_overall", asc=False)
    fast = _safe_top(lat, "avg_latency_ms", asc=True)
    top_lang = _safe_top(lsum, "avg_overall", asc=False)
    bottom_lang = _safe_top(lsum, "avg_overall", asc=True)
    top_q = _safe_top(qsum, "avg_overall", asc=False)
    low_q = _safe_top(qsum, "avg_overall", asc=True)

    ctx = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": str(best["model_name"]) if best is not None else "N/A",
        "best_score": f"{best['avg_overall']:.3f}" if best is not None else "N/A",
        "fast_model": str(fast["model_name"]) if fast is not None else "N/A",
        "fast_latency": f"{fast['avg_latency_ms']:.0f} ms" if fast is not None else "N/A",
        "top_lang": str(top_lang["language"]) if top_lang is not None else "N/A",
        "top_lang_score": f"{top_lang['avg_overall']:.3f}" if top_lang is not None else "N/A",
        "bottom_lang": str(bottom_lang["language"]) if bottom_lang is not None else "N/A",
        "bottom_lang_score": f"{bottom_lang['avg_overall']:.3f}" if bottom_lang is not None else "N/A",
        "top_q": str(top_q["base_qid"]) if top_q is not None else "N/A",
        "top_q_score": f"{top_q['avg_overall']:.3f}" if top_q is not None else "N/A",
        "low_q": str(low_q["base_qid"]) if low_q is not None else "N/A",
        "low_q_score": f"{low_q['avg_overall']:.3f}" if low_q is not None else "N/A",
        "input_mean": f"{tmodel['input_tokens_est_mid_mean'].mean():.1f}" if (not tmodel.empty and 'input_tokens_est_mid_mean' in tmodel.columns) else "N/A",
        "resp_mean": f"{tmodel['response_tokens_est_mid_mean'].mean():.1f}" if (not tmodel.empty and 'response_tokens_est_mid_mean' in tmodel.columns) else "N/A",
        "resp_p90": f"{tmodel['response_tokens_est_mid_p90'].max():.1f}" if (not tmodel.empty and 'response_tokens_est_mid_p90' in tmodel.columns) else "N/A",
        "total_max": f"{int(tmodel['total_tokens_est_mid_max'].max())}" if (not tmodel.empty and 'total_tokens_est_mid_max' in tmodel.columns) else "N/A",
        "model_table": _md_table(msum.sort_values("avg_overall", ascending=False), ["model_name", "n", "avg_overall", "std_overall"], 10),
        "lang_table": _md_table(lsum.sort_values("avg_overall", ascending=False), ["language", "n", "avg_overall", "std_overall"], 24),
        "question_table": _md_table(qsum.sort_values("avg_overall", ascending=False), ["base_qid", "n", "avg_overall", "avg_factual", "avg_completeness", "avg_fluency"], 10),
        "token_model_table": _md_table(tmodel.sort_values("model_name"), ["model_name", "n", "input_tokens_est_mid_mean", "response_tokens_est_mid_mean", "response_tokens_est_mid_p90", "total_tokens_est_mid_max", "remaining_output_tokens_est_min", "remaining_output_tokens_est_max"], 20),
        "token_q_table": _md_table(tq.sort_values("base_question"), ["base_question", "context_tokens_est", "question_tokens_est_avg", "prompt_tokens_est_avg"], 10),
        "token_q_table_extended": _md_table(tq.sort_values("base_question"), ["base_question", "context_tokens_est", "question_tokens_est_min", "question_tokens_est_avg", "question_tokens_est_max", "prompt_tokens_est_min", "prompt_tokens_est_avg", "prompt_tokens_est_max"], 20),
        "token_lq_table": _md_table(tlq.sort_values(["language", "base_question"]), ["language", "base_question", "input_tokens_est_mid", "response_tokens_est_mid_mean", "response_tokens_est_mid_p90", "response_tokens_est_mid_max", "total_tokens_est_mid_mean", "total_tokens_est_mid_max"], 200),
        "token_model_output_budget_table": _md_table(tmodel_out.sort_values("model_name"), ["model_name", "max_model_len", "remaining_output_tokens_est_min", "remaining_output_tokens_est_avg", "remaining_output_tokens_est_max", "effective_output_cap_min", "effective_output_cap_max"], 30),
        "token_row_count": str(len(trow)),
    }

    _write(rp["guide_md"], _render_guide(ctx))
    _write(rp["deep_dive_md"], _render_deep_dive(ctx))
    _write(rp["faq_md"], _render_faq(ctx))
    return True, f"generated: {run_dir}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate VRAM/concurrency/FAQ markdown docs for one run or all runs.")
    p.add_argument("--run-dir", type=Path, help="Specific run directory (e.g., results/runs/a40/<run_id>).")
    p.add_argument("--all-runs", action="store_true", help="Process all runs under results/runs/*/* (default when --run-dir is omitted).")
    p.add_argument("--force", action="store_true", help="Regenerate even when output artifacts already exist.")
    return p.parse_args()


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
