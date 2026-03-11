# LLM Evaluation Framework - Current Context Pipeline

## Scope
This document describes the **active GPU context-evaluation pipeline** and its scoring framework.

Source of questions/context:
- `translations/eu_24_languages_euf_context.py`

Execution/scoring scripts:
- `gpu_runtime/evaluate_context.py`
- `gpu_runtime/evaluate_context_results.py`

Note:
- This document is about the GPU context pipeline.
- The legacy FastAPI app has been removed from the supported workflow.

## Dataset Design

- Languages: 24 EU languages (`BG, HR, CS, DA, NL, EN, ET, FI, FR, DE, EL, HU, GA, IT, LV, LT, MT, PL, PT, RO, SK, SL, ES, SV`)
- Question families: 5 (`Q1..Q5`)
- Context per question: 5 English context entries
- Runs per question-language pair: 3

Per model response count:
- `24 languages x 5 questions x 3 runs = 360 responses`

## Active Scoring Metrics (Context Profile)

Metric outputs written to the scores DB:

- `relevance`
- `factual_accuracy`
- `completeness`
- `fluency`
- `coherence`
- `prompt_alignment`
- `token_efficiency`
- `overall_quality`

### Composite score weights
Using `metrics/metrics_config.yaml` profile `context`:

- relevance: `0.28`
- factual_accuracy: `0.28`
- completeness: `0.20`
- fluency: `0.14`
- coherence: `0.05`
- prompt_alignment: `0.00` (disabled in this profile)
- token_efficiency: `0.05` (enabled in this profile)

Composite formula:

`overall_quality = 0.28*relevance + 0.28*factual_accuracy + 0.20*completeness + 0.14*fluency + 0.05*coherence + 0.05*token_efficiency`

## Storage Schema (Current)

### Results DB (`evaluation_results_euf_context.db`)
Table: `evaluations`

- `id` (INTEGER PK)
- `model_name` (TEXT)
- `language` (TEXT)
- `question_id` (TEXT, e.g. `Q3_DE`)
- `run_number` (INTEGER)
- `question_text` (TEXT)
- `context` (TEXT, JSON serialized)
- `response` (TEXT)
- `timestamp` (TEXT)
- `latency_ms` (REAL)

### Scores DB (`evaluation_scores_euf_context.db`)
Table: `scores`

- `id` (INTEGER PK)
- `evaluation_id` (INTEGER)
- `model_name` (TEXT)
- `language` (TEXT)
- `question_id` (TEXT)
- `relevance` (REAL)
- `factual_accuracy` (REAL)
- `completeness` (REAL)
- `fluency` (REAL)
- `coherence` (REAL)
- `prompt_alignment` (REAL)
- `token_efficiency` (REAL)
- `overall_quality` (REAL)
- `timestamp` (TEXT)

## Important Notes

- `prompt_alignment` and `token_efficiency` are still stored in outputs for schema consistency.
- In the active context profile, `prompt_alignment` contributes `0.00` and `token_efficiency` contributes `0.05` to `overall_quality`.
- The legacy `data/evaluation_questions.json` format from older generic tasks is deprecated in favor of the multilingual context source module.
