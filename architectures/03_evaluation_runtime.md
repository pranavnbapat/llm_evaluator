# Evaluation Runtime in Detail

The evaluation runtime consists of three entry points that share a common infrastructure pattern but handle different data types.

---

## Shared Infrastructure

All three evaluation paths use these core classes:

### `VLlmManager`

Manages the vLLM server lifecycle:

```
start(model_config):
  1. pkill stale vLLM processes
  2. Build vLLM CLI command:
     vllm serve <local_path>
       --host 0.0.0.0 --port 8000
       --tensor-parallel-size 1
       --dtype <float16|bfloat16>
       --max-model-len <N>
       --gpu-memory-utilization <0.75-0.90>
       --served-model-name <name>
       [+ model-specific flags]
  3. Launch subprocess with log file /tmp/vllm_<model>.log
  4. Poll /health until 200 OK (timeout: 30 min)
  5. Verify /v1/models returns expected model_id

stop():
  1. SIGTERM to process group
  2. Wait 10s
  3. SIGKILL if still alive
  4. 3s cooldown
```

**Chat API fallback:** If `/v1/chat/completions` fails (e.g., missing chat template), the text evaluator falls back to `/v1/completions` with a plain prompt string. Vision evaluators have no fallback — plain completions cannot accept images.

### `GPUMonitor`

A background daemon thread that logs once per second:

| Column | Source |
|---|---|
| `timestamp` | System clock |
| `gpu_index`, `util_gpu_pct`, `util_mem_pct` | `nvidia-smi` |
| `mem_total_mb`, `mem_used_mb`, `mem_free_mb` | `nvidia-smi` |
| `temp_c`, `power_w` | `nvidia-smi` |
| `cpu_util_pct` | `/proc/stat` (delta calculation) |
| `ram_total_mb`, `ram_used_mb`, `ram_free_mb` | `/proc/meminfo` |
| `phase`, `model_name`, `eval_language`, ... | Runtime context tags |

The monitor receives context updates via `set_context()` so every row is tagged with the current evaluation phase.

---

## Path A: Context Evaluation (`evaluate_context.py`)

### Data Source

```python
from translations.eu_24_languages_euf_context import get_all_questions_with_context

questions = get_all_questions_with_context()
# 5 questions × 24 EU languages = 120 question instances
```

Each question includes:
- `question_id` (e.g., `Q1`, `Q2`)
- `language` (e.g., `FR`, `DE`)
- `question_text` (localized)
- `context` — a list of search result documents with `title`, `description`, etc.

### Prompt Structure

```
SEARCH RESULTS (in English):
[1] Title: Description...
[2] Title: Description...

FARMER'S QUESTION (in FR):
Comment gérer l'irrigation...?

IMPORTANT INSTRUCTIONS:
1. Answer in the SAME LANGUAGE as the farmer's question (FR)
2. Provide a COMPREHENSIVE but CONCISE answer (2-4 paragraphs)
3. Use SPECIFIC details from the search results
4. Give PRACTICAL, actionable advice that farmers can implement
5. If the search results don't fully answer the question, provide your best expert knowledge

Your response:
```

### Token Budget Enforcement

If `usable_input_tokens` is set in the model config, the prompt is clipped to fit:

```python
# If full prompt exceeds budget
if estimate_tokens(prompt) > usable_input_tokens:
    # Reserve budget for question + instructions
    no_context_tokens = estimate_tokens(prompt_without_context)
    context_budget = max(0, usable_input_tokens - no_context_tokens)
    # Clip context to budget
    clipped_context = clip_text_to_token_budget(context_str, context_budget)
    # Rebuild prompt with clipped context
```

### Database Schema

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    language TEXT,
    question_id TEXT,
    run_number INTEGER,
    question_text TEXT,
    context TEXT,        -- JSON string
    response TEXT,
    timestamp TEXT,
    latency_ms REAL
);
```

---

## Path B: Vision Evaluation (`evaluate_vision.py`)

### Data Source

```python
# data/evaluation_vision_questions.json
tasks, asset_roots = load_vision_dataset(Path("data/evaluation_vision_questions.json"))
```

Supported task types:

| Modality | Task | Description |
|---|---|---|
| `image` | `qa` | Answer a question about an image |
| `image` | `summary` | Summarize the content of an image |
| `pdf` | `qa` | Answer a question about a PDF document |
| `pdf` | `summary` | Summarize a PDF document |

### Task Expansion

The dataset is expanded per-language. If there are 11 base tasks and 24 EU languages, that becomes **264 tasks per model**.

### Image Tasks

1. Resolve image references (local paths or URLs)
2. Encode as base64 data URLs
3. Build OpenAI-style multimodal message:
   ```json
   [
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
     {"type": "text", "text": "You are reviewing attached images..."}
   ]
   ```
4. POST to `/v1/chat/completions`

### PDF Tasks — Map/Reduce Architecture

```
PDF (e.g., 10 pages)
    │
    ▼
render_pdf_to_pngs() ──► page-01.png ... page-10.png
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  MAP PHASE (chunked by pages_per_chunk, default 3)          │
│  ────────────────────────────────────────                   │
│  Batch 1: pages 1-3  ──► VLM ──► partial summary/evidence  │
│  Batch 2: pages 4-6  ──► VLM ──► partial summary/evidence  │
│  Batch 3: pages 7-9  ──► VLM ──► partial summary/evidence  │
│  Batch 4: page 10    ──► VLM ──► partial summary/evidence  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  REDUCE PHASE (if >1 batch)                                 │
│  ───────────────────────                                    │
│  "Synthesize these partial summaries into one coherent..."  │
│  ──► VLM (text-only) ──► final answer/summary              │
└─────────────────────────────────────────────────────────────┘
```

**Auto-recovery:** If a chunk fails with `"max_tokens must be at least 1"` (meaning the chunk was too large for the model's context), the system retries with `pages_per_chunk=1`.

### Vision Database Schema

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    language TEXT,
    question_id TEXT,
    item_id TEXT,
    task_type TEXT,       -- "qa" | "summary"
    modality TEXT,        -- "image" | "pdf"
    run_number INTEGER,
    question_text TEXT,
    context TEXT,
    media_ref TEXT,       -- image path or PDF path
    media_count INTEGER,  -- number of images/pages
    response TEXT,
    timestamp TEXT,
    latency_ms REAL,
    metadata_json TEXT    -- step info, chunk counts, etc.
);

CREATE TABLE evaluation_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER,
    step_type TEXT,       -- "map" | "reduce"
    chunk_index INTEGER,
    page_start INTEGER,
    page_end INTEGER,
    prompt_text TEXT,
    response_text TEXT,
    latency_ms REAL,
    status TEXT,
    created_at TEXT
);
```

---

## Path C: PDF Summary Evaluation (`evaluate_pdf_summary.py`)

A dedicated pipeline for PDF map-reduce summarization benchmarking.

### Map Prompt

```
You are reading pages {start}-{end} of {total} of the document '{file_name}'.
Read the {N} attached page image(s) and write a self-contained summary
of what these pages cover in 4-6 sentences. Capture key facts, headings,
figures, and any actionable items. Do NOT speculate about pages you have not seen.
```

### Reduce Prompt

```
You have been given partial summaries for the document '{file_name}'
({total_pages} pages total), produced by reading pages in batches.

PARTIAL SUMMARIES:
[batch 1] ...
[batch 2] ...

Synthesise these into ONE coherent overall summary of the document in 8-12 sentences.
Cover the document's purpose, structure, and main points.
Do not invent content that is not supported by the partial summaries above.
```

### Database Schema

```sql
CREATE TABLE pdf_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT,
    file_name TEXT,
    total_pages INTEGER,
    batch_size INTEGER,
    num_batches INTEGER,
    overall_summary TEXT,
    overall_summary_time_ms REAL,
    map_total_time_ms REAL,
    total_time_ms REAL,
    started_at TEXT,
    finished_at TEXT,
    status TEXT          -- "ok" | "partial" | "all_batches_failed"
);

CREATE TABLE pdf_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES pdf_runs(id),
    batch_index INTEGER,
    page_start INTEGER,
    page_end INTEGER,
    num_pages INTEGER,
    summary TEXT,
    batch_time_ms REAL,
    status TEXT          -- "ok" | "failed"
);
```
