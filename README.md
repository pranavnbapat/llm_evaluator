# LLM Evaluator

Evaluate configurable LLM sets across 24 EU languages using reproducible metrics.

This repository's supported workflow is the GPU context-evaluation pipeline in `gpu_runtime/`.
The legacy FastAPI app has been removed.

## Documentation

- Deployment and execution on a GPU server:
  [gpu_runtime/README.md](gpu_runtime/README.md)
- Scoring-only setup and usage:
  [README_SCORING.md](README_SCORING.md)

---

## 📊 What This Evaluates

### Models
The model set is configurable and not fixed in this README.

Source of truth:
- Candidate repos: `gpu_runtime/model_repos.txt`
- Active run config: `gpu_runtime/config.yaml` (`models:` block, often generated via `generate_gpu_config.py`)

### 24 EU Languages
BG, HR, CS, DA, NL, EN, ET, FI, FR, DE, EL, HU, GA, IT, LV, LT, MT, PL, PT, RO, SK, SL, ES, SV

### 5 Evaluation Question Families
1. **Organic Weed Control**
2. **Soil Health Restoration**
3. **Climate Adaptation**
4. **EU Funding for Agroecology**
5. **Integrated Pest Management**

### Quality Metrics
- Relevance (semantic similarity)
- Factual Accuracy (NLI-based)
- Completeness (checklist-based)
- Fluency (perplexity + grammar)
- Coherence (discourse flow)
- Prompt Alignment (hallucination detection)
- Token Efficiency

---

## 🔬 Scientific Framework

### Evaluation Pipeline
```
Questions × Languages × Models × Runs → Responses → Metrics → Analysis
```

### Benchmark Protocol
- Fixed multilingual question/context set from `translations/eu_24_languages_euf_context.py`
- Repeated runs controlled by `gpu_runtime/config.yaml`:
  - `evaluation.num_runs`
  - `evaluation.temperature`
  - `evaluation.max_tokens`
- Per-run isolation via `run_id` and run folders under:
  - `results/runs/<gpu_bucket>/<run_id>/`
- Raw benchmark artifacts:
  - `raw/evaluation_results_euf_context.db`
  - per-model JSON summaries in `raw/`
  - `logs/gpu_metrics.csv`
  - `metadata/run_info.json` and `metadata/model_status.json`

### Quality Scoring
The scoring stage (`gpu_runtime/evaluate_context_results.py`) computes:
- Relevance
- Factual Accuracy
- Completeness
- Fluency
- Coherence
- Prompt Alignment
- Token Efficiency
- Overall Quality (weighted composite)

The active metric profile and weights come from:
- `metrics/metrics_config.yaml` (context profile used by default in scoring script)

### Statistical Analysis Utilities
The repo includes statistical helper functions in:
- `metrics/statistical_analysis.py` (ICC, paired t-test, Cohen's d, bootstrap CI, cross-language consistency)

Use these on scored outputs (`scores/evaluation_scores_euf_context.db`) when you need formal model comparison.

---

## 📈 Results

After evaluation, you get:
- **Raw SQLite database** (`evaluation_results_euf_context.db`) - responses/latency
- **Scored SQLite database** (`evaluation_scores_euf_context.db`) - per-response quality metrics
- **JSON summaries** - per-model completion/success info

Query the database:
```bash
sqlite3 results/runs/<gpu_bucket>/<run_id>/raw/evaluation_results_euf_context.db \
  "SELECT model_name, AVG(latency_ms) FROM evaluations GROUP BY model_name;"
```

---

## 🔬 Evaluation Method

All metrics are computed **locally** on GPU after response generation:

| Metric | Method |
|--------|--------|
| Relevance | Sentence-transformers embeddings |
| Factual Accuracy | NLI/context entailment |
| Completeness | Context coverage scoring |
| Fluency | Zero-shot classification |
| Coherence | Zero-shot classification |
| Prompt Alignment | Semantic alignment |

---
