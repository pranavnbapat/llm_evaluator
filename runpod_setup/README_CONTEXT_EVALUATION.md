# Context-Based Evaluation Deployment Guide

This guide explains how to deploy the new RAG (Retrieval-Augmented Generation) evaluation with search result context.

## What's New

1. **New questions file**: `translations/eu_24_languages_euf_context.py`
   - 5 questions in all 24 EU languages
   - Each question includes 5 English context entries (search results)
   - Total: 120 questions with context

2. **New evaluation script**: `runpod_setup/evaluate_context.py`
   - Evaluates models with context-enhanced prompts
   - Saves to run-based folder under `results/runs/<gpu_bucket>/<run_id>/raw/`

3. **New scoring script**: `runpod_setup/evaluate_context_results.py`
   - Scores context-based responses
   - Saves to run-based folder under `results/runs/<gpu_bucket>/<run_id>/scores/`

## Deployment Steps

### 1. Push Changes to GitHub

```bash
# From project root
git add translations/eu_24_languages_euf_context.py
git add runpod_setup/evaluate_context.py
git add runpod_setup/evaluate_context_results.py
git add runpod_setup/README_CONTEXT_EVALUATION.md
git commit -m "Add context-based RAG evaluation framework"
git push
```

### 2. On RunPod Instance

```bash
# SSH into RunPod
ssh root@your-runpod-ip

# Navigate to project
cd /workspace/llm_evaluator

# Pull latest changes
git pull
```

### 3. Setup Environment

```bash
# Run setup (if not already done)
cd runpod_setup
bash setup.sh

# Download models (if not already done)
python download_models.py
```

### 4. Run Context-Based Evaluation

```bash
# Make sure you're in runpod_setup directory
cd /workspace/llm_evaluator/runpod_setup

# Set API key
export OPENAI_API_KEY=your_key_here

# Run evaluation
python evaluate_context.py
```

This will:
- Start each model in vLLM
- Send questions WITH context to the model
- Auto-detect GPU bucket (or use `EVAL_RUN_GPU` override)
- Save responses to `../results/runs/<gpu_bucket>/<run_id>/raw/evaluation_results_euf_context.db`
- Save JSON summaries

### 5. Score Responses

```bash
# From project root
cd /workspace/llm_evaluator

# Run scoring
python runpod_setup/evaluate_context_results.py
```

This will:
- Read from `evaluation_results_euf_context.db`
- Compute 7 quality metrics for each response
- Save to `evaluation_scores_euf_context.db` in the same run folder (`scores/`)

### 6. Export Results

```bash
# Export to Excel
python sqlite_to_excel.py
```

## File Structure

```
llm_evaluator/
├── translations/
│   ├── eu_24_languages_euf_context.py      # NEW: Questions + context
│   └── eu_24_languages.py                   # OLD: Original questions
├── runpod_setup/
│   ├── evaluate_context.py                  # NEW: Context evaluation
│   ├── evaluate_context_results.py          # NEW: Context scoring
│   ├── evaluate.py                          # OLD: Original evaluation
│   └── README_CONTEXT_EVALUATION.md         # This file
├── evaluate_results.py                      # OLD: Original scoring
└── results/
    ├── runs/
    │   └── <gpu_bucket>/
    │       └── <run_id>/
    │           ├── raw/                     # results DB/XLSX + model JSON summaries
    │           ├── scores/                  # scores DB/XLSX
    │           ├── logs/                    # evaluate + scoring logs + gpu_metrics.csv
    │           ├── insights/                # generated reports/charts/data
    │           └── metadata/                # run metadata
    └── latest/
        └── <gpu_bucket> -> ../runs/<gpu_bucket>/<run_id>
```

## Key Differences from Original Evaluation

| Aspect | Original | Context-Based |
|--------|----------|---------------|
| Questions | 120 (5×24) | 120 (5×24) |
| Context | None | 5 search results per question |
| Database | `evaluation_results.db` | `evaluation_results_euf_context.db` |
| Scores DB | `evaluation_scores.db` | `evaluation_scores_euf_context.db` |
| Prompt | Simple question | RAG-style with context |
| Evaluation | General metrics | Context-aware metrics |

## Context Format

Each question now includes context like this:

```python
{
    "question_id": "Q1_BG",
    "language": "BG",
    "question": "Какви органични методи...",
    "context": [
        {
            "title": "Innovating Together for Emission-Free Weed Control...",
            "subtitle": "Reducing herbicide use...",
            "description": "Reduced crop protection product use...",
            "keywords": ["emission reduction", "crop protection", ...],
            "ko_content_flat": ["This project..."]
        },
        # ... 4 more entries
    ]
}
```

## Troubleshooting

### Run-folder controls (optional)

```bash
# force GPU bucket mapping
export EVAL_RUN_GPU=a40

# explicit run id
export EVAL_RUN_ID=2026-03-01_220000_context_eval

# explicit run directory (highest priority)
export EVAL_RUN_DIR=/workspace/llm_evaluator/results/runs/a40/2026-03-01_220000_context_eval
```

### Database locked error
```bash
# Check if another process is using the database
lsof results/runs/<gpu_bucket>/<run_id>/raw/evaluation_results_euf_context.db

# If stuck, restart the evaluation
rm results/runs/<gpu_bucket>/<run_id>/raw/evaluation_results_euf_context.db
python runpod_setup/evaluate_context.py
```

### Out of memory
```bash
# Reduce GPU memory utilization in config.yaml
# Change gpu_memory_util from 0.95 to 0.85
```

### vLLM won't start
```bash
# Check logs
vllm serve /path/to/model --host 0.0.0.0 --port 8000

# Or check if port is in use
lsof -i :8000
kill <PID>
```

## Expected Runtime

With 5 models × 120 questions × 3 runs = 1,800 responses:
- Evaluation: ~4-6 hours (depending on model speed)
- Scoring: ~30-60 minutes
- Total: ~5-7 hours

## Questions?

- Check logs in `results/` directory
- Review `evaluate_context.py` for evaluation logic
- Review `runpod_setup/evaluate_context_results.py` for scoring logic
