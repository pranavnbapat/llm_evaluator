# Key Design Decisions

This document explains the rationale behind critical architectural choices in the `gpu_runtime/` evaluation system.

---

## 1. One Model at a Time in vLLM

**Decision:** Only ONE model is loaded in vLLM at a time. The loop is:
```
Start vLLM → Evaluate all tasks → Stop vLLM → Cooldown → Next model
```

**Rationale:**
- vLLM holds the full model weights + KV cache in VRAM
- Loading 2-3 models simultaneously would cause GPU OOM on all but the largest GPUs
- Python-level model swapping would fragment CUDA memory and leak allocator state
- Cold-start overhead is acceptable because evaluation runs are long-running batch jobs

**Trade-off:** ~30-120 seconds of startup time per model, but guaranteed stability across 10+ model evaluation campaigns.

---

## 2. SQLite + WAL Mode

**Decision:** All raw responses and scores are stored in SQLite databases with Write-Ahead Logging (WAL) enabled.

**Rationale:**
- Zero setup — no database server required
- WAL mode allows readers during writes (scoring can read while evaluation is still running)
- Survives crashes without corruption
- Easy to export to Excel/Pandas via `sqlite3` + `pandas.read_sql_query()`
- Single-file portability for moving runs between servers

```python
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
```

**Trade-off:** Not suited for distributed concurrent writers, but evaluation is single-process by design.

---

## 3. Run Isolation with Symlinks

**Decision:** Every execution creates a timestamped `run_id` under `results/runs/<gpu_bucket>/`. A `results/latest/<gpu_bucket>` symlink always points to the newest run.

**Rationale:**
- Runs are immutable — old data is never overwritten
- Downstream scripts can always find the latest data without hardcoding run IDs
- Easy to compare historical runs (same GPU bucket, different dates)
- Symlinks avoid copying data

```
results/runs/a40/2026-03-07_120000_context_eval/
results/runs/a40/2026-03-14_090000_context_eval/
results/latest/a40 -> ../runs/a40/2026-03-14_090000_context_eval
```

---

## 4. Dual DB Pattern (Raw + Scores)

**Decision:** Raw responses go into `evaluation_results_*` DBs first. Scoring is a separate pass that reads the raw DB and writes to `evaluation_scores_*` DBs.

**Rationale:**
- Allows re-scoring with different metric weights without re-running expensive inference
- Raw DB preserves the exact model outputs for audit/debugging
- Score DB can be regenerated, truncated, or experimented with independently
- Separation of concerns: evaluation = inference, scoring = offline analysis

---

## 5. Batch Scoring

**Decision:** The scoring scripts use batched inference for fluency, coherence, and NLI metrics.

**Rationale:**
- Running transformer pipelines one-by-one would be ~100× slower
- Batch size 96-128 keeps the GPU fully utilized during scoring
- Pre-compute expensive metrics once, then assemble per-item scores

```python
# Pre-compute once per batch
fluency_scores   = evaluator.calculate_fluency_batch(responses, languages)
coherence_scores = evaluator.calculate_coherence_batch(responses)
nli_scores       = evaluator.calculate_nli_entailment_batch(responses, contexts)

# Assemble per item
for idx, item in enumerate(batch):
    scores = evaluator.evaluate_response(
        ...,
        precomputed_scores={
            "fluency": fluency_scores[idx],
            "coherence": coherence_scores[idx],
            "factual_accuracy": nli_scores[idx],
        }
    )
```

---

## 6. Token Budget Clipping

**Decision:** Prompts are dynamically clipped to fit within `usable_input_tokens`.

**Rationale:**
- Prevents context window overflow errors at inference time
- Maximizes usable context while guaranteeing room for the answer
- Context is trimmed before the question/instructions, so the core query is always preserved

```python
usable_input_tokens = max_model_len - max_output_tokens

if estimate_tokens(prompt) > usable_input_tokens:
    # Trim context, never the question or instructions
    context_budget = usable_input_tokens - no_context_tokens
    clipped_context = clip_text_to_token_budget(context_str, context_budget)
```

---

## 7. Background Shell Wrappers

**Decision:** `run_*_background.sh` scripts wrap Python evaluators in `tmux` or `nohup`.

**Rationale:**
- Evaluation runs take hours (10 models × 360 questions × latency)
- SSH disconnects must not kill the run
- `tmux` allows attaching to inspect progress live
- `nohup` is the fallback when tmux is unavailable

```bash
# tmux mode (default)
tmux new-session -d -s eval_context \
  "python gpu_runtime/evaluate_context.py 2>&1 | tee logs/evaluate_context.log"

# Attach anytime
tmux attach -t eval_context
```

---

## 8. GPU Bucket Auto-Detection

**Decision:** Every script reads `nvidia-smi` to auto-classify runs into buckets.

**Rationale:**
- Eliminates manual configuration on every run
- Ensures results are organized by actual hardware, not user memory
- Allows the same codebase to run on A40, L40S, A100, H200, B200 without code changes

```python
def detect_gpu_bucket() -> tuple[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True
    )
    name = result.stdout.strip().lower()
    if "b200" in name: return "b200", f"nvidia-smi:{name}"
    if "h200" in name: return "h200_sxm", f"nvidia-smi:{name}"
    if "l40s" in name: return "l40s", f"nvidia-smi:{name}"
    # ...
```

**Override:** `EVAL_RUN_GPU=a40` forces a bucket for testing or cross-bucket scoring.

---

## 9. Static VRAM Sizing Before Download

**Decision:** `runtime_common/model_static_check.py` estimates GPU fit using only `config.json` metadata, before any weights are downloaded.

**Rationale:**
- Downloading a 70B model (~140GB) only to find it doesn't fit is wasteful
- Checkpoint size can be fetched from `safetensors.index.json` metadata (no weight download)
- Config-based heuristics are accurate enough for pass/fit classification
- The `--allow-fits` filter prevents un-runnable models from entering the config

---

## 10. PDF Map/Reduce for Vision Models

**Decision:** PDF documents are rendered to PNGs and processed in chunked map/reduce passes.

**Rationale:**
- Vision-LMs cannot natively read PDF bytes
- `pdftoppm` (Poppler) is fast, deterministic, and server-friendly
- Map/reduce allows processing arbitrarily long documents within context limits
- Single-page fallback auto-recovers from chunk-too-large errors

```
PDF → page-01.png ... page-N.png
→ [chunk 1] → VLM → partial summary
→ [chunk 2] → VLM → partial summary
→ reduce(all partials) → VLM → final summary
```
