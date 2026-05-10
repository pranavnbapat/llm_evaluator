# End-to-End Data Flow Walkthrough

This document walks through exactly what happens when you run a full context evaluation on an `l40s` GPU.

---

## Step 1: Configuration

```bash
$ python gpu_runtime/generate_gpu_config.py l40s \
    --repos-file gpu_runtime/model_repos.txt \
    --concurrent-users 50 \
    --target-max-output-tokens 512
```

**What happens:**
1. Reads `model_repos.txt` — one HuggingFace repo ID per line
2. For each repo:
   - Fetches `config.json` from HuggingFace
   - Fetches `model.safetensors.index.json` to get real checkpoint size (if available)
   - Extracts dimensions: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`
   - Computes KV cache size for candidate `seq_lens`: `[4096, 8192, 16384]`
   - Checks concurrent-user cap: can 50 users fit at this seq_len?
   - Classifies fit: `comfortable`, `tight`, `very tight`, or `unlikely`
3. Rewrites `gpu_runtime/config.yaml`:
   - Replaces the `models:` block with only models that pass `--allow-fits`
   - Updates `generation_profile:` with GPU, concurrent users, and timestamp
   - Sets `evaluation.max_tokens` if `--target-max-output-tokens` was provided

**Example output in config.yaml:**
```yaml
generation_profile:
  gpu: "l40s"
  concurrent_users: 50
  target_max_output_tokens: 512
  seq_lens: "4096,8192,16384"
  allow_fits: "comfortable"
  generated_at: "2026-05-09 18:53:12"

models:
  eurollm_9b_instruct_2512_fp16_l40s:
    name: "eurollm-9b-instruct-2512-l40s"
    repo: "utter-project/EuroLLM-9B-Instruct-2512"
    local_path: "/workspace/models/eurollm_9b_instruct_2512_fp16_l40s"
    quant: null
    dtype: "float16"
    max_model_len: 4096
    usable_input_tokens: 3072
    gpu_memory_util: 0.90
```

---

## Step 2: Model Download

```bash
$ python gpu_runtime/download_models.py
```

**What happens:**
1. Reads `config.yaml` `models:` block
2. For each model:
   - Calls `huggingface_hub.snapshot_download(repo_id, local_dir=local_path, token=HF_TOKEN)`
   - Resumes interrupted downloads
   - Reports final size in GB
3. Stores all models in `/workspace/models/<model_key>/`

---

## Step 3: Evaluation

```bash
$ python gpu_runtime/evaluate_context.py
```

**What happens:**

### 3a. Run Directory Setup

```
results/runs/l40s/2026-05-09_185312_context_eval/
├── raw/
├── scores/
├── logs/
├── insights/
└── metadata/
```

Also updates symlink:
```
results/latest/l40s -> ../runs/l40s/2026-05-09_185312_context_eval
```

### 3b. GPU Monitor Start

`GPUMonitor` starts a background thread writing to `logs/gpu_metrics.csv` every second.

### 3c. Per-Model Loop

**Model 1: `eurollm-9b-instruct-2512-l40s`**

```
1. VLlmManager.start(model_config)
   → Build: vllm serve /workspace/models/eurollm_9b_instruct_2512_fp16_l40s
            --host 0.0.0.0 --port 8000
            --dtype float16
            --max-model-len 4096
            --gpu-memory-utilization 0.90
   → Wait for /health = 200 (takes ~30-120s)
   → Verify model_id matches

2. Evaluator.evaluate_model()
   → Load 120 questions (5 × 24 languages)
   → FOR each question:
        FOR run in 1..3:
          Build prompt with context
          POST /v1/chat/completions
          Record: response, latency_ms
          INSERT INTO evaluation_results_euf_context.db

3. VLlmManager.stop()
   → SIGTERM vLLM process
   → 3s cooldown
```

**Model 2, 3, 4, ...** — repeat until all models evaluated.

### 3d. Finalization

- Export SQLite → `evaluation_results_euf_context.xlsx`
- Export per-model sheets → `evaluation_results_euf_context_by_model.xlsx`
- Write `metadata/model_status.json`:
  ```json
  [
    {
      "model_name": "eurollm-9b-instruct-2512-l40s",
      "repo": "utter-project/EuroLLM-9B-Instruct-2512",
      "status": "evaluated",
      "started_at": "2026-05-09T18:53:12",
      "finished_at": "2026-05-09T19:12:34",
      "details": "successful_responses=360/360"
    }
  ]
  ```

---

## Step 4: Scoring

```bash
$ python gpu_runtime/evaluate_context_results.py
```

**What happens:**

1. **Resolve paths** via `EVAL_RUN_DIR` → `results/runs/l40s/2026-05-09_185312_context_eval/`
2. **Load models:**
   - SentenceTransformer (`paraphrase-multilingual-mpnet-base-v2`) → CUDA
   - NLI pipeline (`cross-encoder/nli-deberta-v3-base`) → CUDA
   - Fluency/Coherence classifiers → CUDA
3. **Read raw DB:** `SELECT * FROM evaluations ORDER BY model_name, language, question_id, run_number`
4. **Batch scoring loop:**
   ```
   FOR batch in batches_of_96:
     fluency    = calculate_fluency_batch(batch_responses, batch_languages)
     coherence  = calculate_coherence_batch(batch_responses)
     nli        = calculate_nli_entailment_batch(batch_responses, batch_contexts)

     FOR idx, item in enumerate(batch):
       scores = evaluate_response(
         question_id=item.question_id,
         response_text=item.response,
         reference_data=item.ref_data,
         precomputed_scores={fluency: ..., coherence: ..., factual_accuracy: ...}
       )
       INSERT INTO scores.db
   ```
5. **Commit** every 200 rows
6. **Export** → `evaluation_scores_euf_context.xlsx`
7. **Summary stats:**
   ```
   eurollm-9b-instruct-2512-l40s: 360 responses, avg quality: 0.782
   granite-4.1-8b-l40s: 360 responses, avg quality: 0.815
   ...
   ```

---

## Step 5: Insights

```bash
$ bash gpu_runtime/run_post_scoring_insights.sh \
    --run-dir results/runs/l40s/2026-05-09_185312_context_eval
```

**What happens:**

1. `generate_context_charts.py`
   ```
   Read: raw/evaluation_results_euf_context.db
         scores/evaluation_scores_euf_context.db
   Write: insights/charts/overall_quality_by_model.png
          insights/charts/language_heatmap.png
          insights/data/model_summary.csv
   ```

2. `generate_presentation_qa.py`
   ```
   Read: raw/*.db + scores/*.db
   Write: insights/Presentation_QA.md
   ```

3. `generate_context_token_budget.py`
   ```
   Read: raw/*.db (question_text, context, response)
   Write: insights/data/token_budget_analysis.csv
   ```

4. `generate_context_vram_docs.py`
   ```
   Read: insights/data/token_budget_*.csv
   Write: insights/data/vram_requirements.md
   ```

5. `gpu_efficiency/generate_gpu_efficiency_report.py`
   ```
   Read: logs/gpu_metrics.csv
   Write: insights/gpu_efficiency/gpu_efficiency_report.md
          insights/gpu_efficiency/gpu_utilization_by_phase.png
   ```

6. `generate_gpu_insights_report.py`
   ```
   Read: All runs under results/runs/l40s/
   Write: insights/GPU_Insights_Report_l40s.md
   ```

---

## Final Output Structure

```
results/runs/l40s/2026-05-09_185312_context_eval/
├── raw/
│   ├── evaluation_results_euf_context.db
│   ├── evaluation_results_euf_context.xlsx
│   ├── evaluation_results_euf_context_by_model.xlsx
│   └── eurollm_9b_instruct_2512_l40s_context_20260509_185312.json
├── scores/
│   ├── evaluation_scores_euf_context.db
│   └── evaluation_scores_euf_context.xlsx
├── logs/
│   ├── evaluate_context_20260509_185312.log
│   ├── evaluate_context_results_20260509_201500.log
│   └── gpu_metrics.csv          ← 1-second granularity GPU telemetry
├── insights/
│   ├── charts/
│   │   ├── overall_quality_by_model.png
│   │   ├── language_heatmap.png
│   │   └── ...
│   ├── data/
│   │   ├── model_summary.csv
│   │   ├── language_summary.csv
│   │   ├── token_budget_analysis.csv
│   │   └── vram_requirements.md
│   ├── Presentation_QA.md
│   ├── GPU_Insights_Report_l40s.md
│   └── gpu_efficiency/
│       ├── gpu_efficiency_report.md
│       └── gpu_utilization_by_phase.png
└── metadata/
    ├── run_info.json            ← run_id, gpu_bucket, paths
    ├── model_status.json        ← per-model pass/fail status
    └── scoring_info.json        ← scoring_db path, source_db path
```
