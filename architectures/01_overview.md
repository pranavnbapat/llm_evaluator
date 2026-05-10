# GPU Runtime — High-Level Architecture Overview

This document describes the model evaluation architecture for the `gpu_runtime/` folder.

---

## Two Evaluation Paths

| Path | Entry Point | Purpose |
|---|---|---|
| **Text / Context** | `generate_gpu_config.py` + `evaluate_context.py` | RAG-style evaluation with search result context across 24 EU languages |
| **Multimodal / Vision** | `generate_gpu_vision_config.py` + `evaluate_vision.py` | Image QA/summary and PDF QA/summary via VLM |

---

## Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT LAYER                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  model_repos.txt          config.yaml          .env (HF_TOKEN)         GPU Server       │
│  (HF repo list)        (generated config)    (secrets/env vars)      (nvidia-smi)       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               SETUP & CONFIGURATION                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  setup.sh ───────► install deps, uv, venv, vLLM, create dirs (/workspace/models, etc.) │
│                                                                                         │
│  generate_gpu_config.py ──► model_static_check.py ──► fetch HF configs, estimate VRAM │
│  (text models)                (weights + KV cache fit)    fit, write config.yaml        │
│                                                                                         │
│  generate_gpu_vision_config.py ──► same flow but for multimodal/vision models          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL DOWNLOAD LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  download_models.py ──► reads config.yaml models: block ──► snapshot_download()        │
│                         downloads each model to /workspace/models/<model_key>/           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            EVALUATION RUNTIME (3 Paths)                                 │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────┐ │
│  │   Path A: TEXT / CONTEXT    │  │   Path B: VISION/MULTIMODAL │  │  Path C: PDF    │ │
│  │                             │  │                             │  │  SUMMARY ONLY   │ │
│  │  evaluate_context.py        │  │  evaluate_vision.py         │  │ evaluate_pdf_   │ │
│  │                             │  │                             │  │ summary.py      │ │
│  └─────────────┬───────────────┘  └─────────────┬───────────────┘  └────────┬────────┘ │
│                │                                │                           │          │
│                ▼                                ▼                           ▼          │
│  ┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────┐   │
│  │  Data Source:           │      │  Data Source:           │    │  Data Source:   │   │
│  │  translations/          │      │  data/evaluation_       │    │  files/*.pdf    │   │
│  │  eu_24_languages_       │      │  vision_questions.json  │    │                 │   │
│  │  euf_context.py         │      │  (images + PDFs)        │    │                 │   │
│  └─────────────┬───────────┘      └─────────────┬───────────┘    └─────────────────┘   │
│                │                                │                                      │
│                ▼                                ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         SHARED INFRASTRUCTURE PER MODEL                          │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────────────┐  │  │
│  │  │ VLlmManager  │───►│   vLLM       │───►│  OpenAI-compatible API (localhost) │  │  │
│  │  │(start/stop)  │    │  Server      │    │  /v1/chat/completions              │  │  │
│  │  └──────────────┘    └──────────────┘    └────────────────────────────────────┘  │  │
│  │         │                                                                         │  │
│  │         ▼                                                                         │  │
│  │  ┌──────────────┐    ┌──────────────┐                                            │  │
│  │  │  GPUMonitor  │───►│ gpu_metrics  │  (1Hz logging: GPU util, VRAM, temp,      │  │
│  │  │(background)  │    │   .csv       │   power, CPU, RAM + eval context tags)    │  │
│  │  └──────────────┘    └──────────────┘                                            │  │
│  │         │                                                                         │  │
│  │         ▼                                                                         │  │
│  │  ┌────────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  EVALUATION LOOP (per model):                                              │  │  │
│  │  │   FOR each model in config.yaml:                                           │  │  │
│  │  │    1. Start vLLM with model_config (dtype, max_model_len, gpu_mem_util)   │  │  │
│  │  │    2. FOR each question/task in dataset:                                   │  │  │
│  │  │       • Build prompt (with context / images / PDF page chunks)             │  │  │
│  │  │       • Call vLLM API (chat_completion or chat_completion_multimodal)     │  │  │
│  │  │       • Record latency, response, metadata                                 │  │  │
│  │  │    3. Stop vLLM, cooldown, next model                                      │  │  │
│  │  └────────────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────────┘  │
│                │                                │                           │          │
│                ▼                                ▼                           ▼          │
│  ┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────┐   │
│  │  Raw Output:            │      │  Raw Output:            │    │  Raw Output:    │   │
│  │  evaluation_results_    │      │  evaluation_results_    │    │ evaluation_pdf_ │   │
│  │  euf_context.db         │      │  euf_vision.db          │    │ summaries.db    │   │
│  │  + per-model JSON       │      │  + evaluation_steps     │    │ + pdf_batches   │   │
│  └─────────────────────────┘      └─────────────────────────┘    └─────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SCORING LAYER                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  evaluate_context_results.py        evaluate_vision_results.py                          │
│         │                                    │                                          │
│         ▼                                    ▼                                          │
│  ┌─────────────────────┐            ┌─────────────────────┐                            │
│  │  Read SQLite DB     │            │  Read SQLite DB     │                            │
│  │  (raw responses)    │            │  (raw responses)    │                            │
│  └──────────┬──────────┘            └──────────┬──────────┘                            │
│             │                                   │                                       │
│             ▼                                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │  metrics/scientific_metrics.py  ──►  ResponseEvaluator                          │   │
│  │  • Embedding similarity (sentence-transformers)                                 │   │
│  │  • NLI entailment for factual accuracy (transformers pipeline)                  │   │
│  │  • Fluency / coherence scoring (batch inference)                                │   │
│  │  • Relevance, completeness, prompt_alignment, token_efficiency                  │   │
│  │  • Overall quality = weighted aggregate                                         │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│             │                                   │                                       │
│             ▼                                   ▼                                       │
│  ┌─────────────────────┐            ┌─────────────────────┐                            │
│  │  Scores Output:     │            │  Scores Output:     │                            │
│  │  evaluation_scores_ │            │  evaluation_scores_ │                            │
│  │  euf_context.db     │            │  euf_vision.db      │                            │
│  │  + .xlsx exports    │            │  + .xlsx exports    │                            │
│  └─────────────────────┘            └─────────────────────┘                            │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               POST-SCORING INSIGHTS                                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  run_post_scoring_insights.sh ──► orchestrates:                                         │
│    • insights/generate_context_charts.py          (metric charts + CSVs)               │
│    • insights/generate_presentation_qa.py         (QA artifacts for presentations)     │
│    • insights/generate_context_token_budget.py    (token budget analysis)              │
│    • insights/generate_context_vram_docs.py       (VRAM/context markdown docs)         │
│    • insights/gpu_efficiency/...                  (GPU utilization efficiency reports) │
│    • insights/generate_gpu_insights_report.py     (combined GPU-level report)          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUT DIRECTORY STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   results/runs/<gpu_bucket>/<run_id>/                                                   │
│   ├── raw/                           ← evaluation DBs, JSON summaries, Excel exports   │
│   ├── scores/                        ← scoring DB, score Excel exports                 │
│   ├── logs/                          ← evaluate/scoring logs, gpu_metrics.csv          │
│   ├── insights/                      ← generated charts, reports, markdown docs        │
│   ├── metadata/                      ← run_info.json, model_status.json, scoring_info  │
│   └── media_pages/                   ← cached PDF page PNG renders (vision only)       │
│                                                                                         │
│   results/latest/<gpu_bucket> ──symlink──► ../runs/<gpu_bucket>/<latest_run_id>         │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Patterns

| Pattern | Description |
|---|---|
| **GPU Bucket Auto-Detection** | Every script reads `nvidia-smi` to auto-classify runs into buckets (`a40`, `l40s`, `a100`, `h200_sxm`, `b200`, etc.). Can be overridden via `EVAL_RUN_GPU`. |
| **Run Isolation** | Each execution creates a timestamped `run_id` under `results/runs/<gpu_bucket>/`, ensuring no overwrites. A `latest/` symlink always points to the newest run. |
| **Model-by-Model vLLM Cycling** | Only ONE model is loaded in vLLM at a time. The loop is: start vLLM → evaluate all tasks → stop vLLM → cooldown → next model. This avoids GPU OOM when evaluating many models. |
| **Dual DB Pattern** | Raw responses go into `evaluation_results_*` DBs first. Scoring is a separate pass that reads the raw DB and writes to `evaluation_scores_*` DBs. |
| **Batch Scoring** | The scoring scripts use batched inference (`EVALUATOR_SCORE_BATCH_SIZE`, default 96) for fluency, coherence, and NLI metrics to maximize GPU utilization. |
| **Background Shell Wrappers** | `run_*_background.sh` scripts wrap the Python evaluators in `tmux` or `nohup` for long-running GPU server jobs. |

---

## Evaluation Paths Summary

| Path | Input | What It Tests | Output DB |
|---|---|---|---|
| **Context/Text** (`evaluate_context.py`) | 5 EU-FarmBook questions × 24 languages + search result context | RAG-style QA quality with retrieved context | `evaluation_results_euf_context.db` |
| **Vision/Multimodal** (`evaluate_vision.py`) | `evaluation_vision_questions.json` with images/PDFs | Image QA, image summary, PDF QA, PDF summary via VLM | `evaluation_results_euf_vision.db` |
| **PDF Summary** (`evaluate_pdf_summary.py`) | `files/*.pdf` | Dedicated PDF map-reduce summarization benchmark | `evaluation_pdf_summaries.db` |
