# Post-Scoring Insights Pipeline

After scoring, `run_post_scoring_insights.sh` transforms DB rows into human-readable artifacts.

Entry point: `gpu_runtime/run_post_scoring_insights.sh`

---

## Usage

```bash
# Single run
bash gpu_runtime/run_post_scoring_insights.sh --run-dir results/runs/a40/2026-03-07_120000_context_eval

# All runs (bulk mode)
bash gpu_runtime/run_post_scoring_insights.sh --all-runs

# Force regenerate existing outputs
bash gpu_runtime/run_post_scoring_insights.sh --run-dir <path> --force
```

---

## Insight Generators

| Script | Reads | Produces |
|---|---|---|
| `generate_context_charts.py` | `scores/*.db` | `insights/charts/*.png` — bar charts, heatmaps, violin plots per model/language/metric |
| `generate_presentation_qa.py` | `scores/*.db` + `raw/*.db` | `insights/Presentation_QA.md` — side-by-side model answers for manual review |
| `generate_context_token_budget.py` | `raw/*.db` (prompt lengths) | `insights/data/token_budget_*.csv` — input/output token statistics |
| `generate_context_vram_docs.py` | token budget CSVs | `insights/data/vram_*.md` — VRAM requirement documentation |
| `gpu_efficiency/generate_gpu_efficiency_report.py` | `logs/gpu_metrics.csv` | `insights/gpu_efficiency/*.png` + `.md` — GPU utilization during eval phases |
| `generate_gpu_insights_report.py` | All above + all runs per GPU bucket | `insights/GPU_Insights_Report_*.md` — aggregate cross-run report |

---

## GPU Efficiency Reporting

The `GPUMonitor` logs a CSV row **every second** with:

| Column | Source |
|---|---|
| `timestamp` | System clock |
| `gpu_index`, `util_gpu_pct`, `util_mem_pct` | `nvidia-smi` |
| `mem_total_mb`, `mem_used_mb`, `mem_free_mb` | `nvidia-smi` |
| `temp_c`, `power_w` | `nvidia-smi` |
| `cpu_util_pct` | `/proc/stat` (delta calculation) |
| `ram_total_mb`, `ram_used_mb`, `ram_free_mb` | `/proc/meminfo` |
| `phase`, `model_name`, `eval_language`, ... | Runtime context tags |

The efficiency report correlates these timestamps with evaluation DB rows to answer:

- **"How efficiently was the GPU used during model loading vs. inference?"**
- **"Which model saturated the GPU?"**
- **"Was there thermal throttling?"**

### Phase Tags

| Phase | Meaning |
|---|---|
| `idle` | Between models, no vLLM running |
| `loading_model` | vLLM startup / weight loading |
| `evaluating` | Active inference (text context path) |
| `map_batch` | PDF page batch inference (vision path) |
| `reduce` | PDF summary synthesis (text-only) |
| `stopping_model` | vLLM teardown |
| `model_start_failed` | vLLM failed to start |

---

## Charts Generated

`generate_context_charts.py` produces:

### Per-Run Charts
- **Model Overall Quality** — bar chart of average `overall_quality` per model
- **Language Heatmap** — heatmap of scores per language × model
- **Metric Breakdown** — grouped bar chart of all 7 metrics per model
- **Latency Distribution** — violin plot of response latencies

### Data Exports
- `insights/data/model_summary.csv`
- `insights/data/language_summary.csv`
- `insights/data/question_summary.csv`

---

## Presentation QA

`generate_presentation_qa.py` creates a markdown file with:

- Side-by-side answers from all evaluated models for each question
- Highlighting of best/worst responses per metric
- Tables for quick manual review

This is used for human-in-the-loop validation of automatic scores.

---

## Token Budget Analysis

`generate_context_token_budget.py` analyzes:

- Average input tokens per question (context + prompt)
- Average output tokens per response
- Token efficiency per model
- Language-specific token usage patterns

This informs future context window sizing decisions.

---

## VRAM Documentation

`generate_context_vram_docs.py` produces markdown docs like:

```markdown
## VRAM Requirements for Context Evaluation

| Model | max_model_len | usable_input_tokens | gpu_memory_util | Fit on L40S (48GB) |
|---|---|---|---|---|
| euro-llm-9b | 4096 | 3072 | 0.90 | Comfortable |
| granite-4.1-8b | 8192 | 7168 | 0.90 | Comfortable |
```

---

## Aggregate GPU Insights Report

`generate_gpu_insights_report.py` scans **all runs** for a GPU bucket and produces a combined report:

- Cross-run model performance trends
- Best-performing models across multiple evaluation dates
- Language consistency scores
- GPU efficiency trends over time

This is the top-level report for stakeholders.
