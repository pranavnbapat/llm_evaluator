# Excel Export Structure (Current Context Evaluation)

This file documents the **actual** Excel outputs currently produced in `results/`.

## 1) `evaluation_results_euf_context.xlsx`

Single sheet: `Sheet1`

| Column | Type | Description |
| --- | --- | --- |
| id | integer | Row id from results DB |
| model_name | string | Model identifier |
| language | string | Language code (24 EU languages) |
| question_id | string | Per-language question id (e.g., `Q2_FR`) |
| run_number | integer | Repeat index (typically 1..3) |
| question_text | text | Prompt question in target language |
| context | text | JSON-serialized context entries |
| response | text | Model response text |
| timestamp | datetime/text | Generation timestamp |
| latency_ms | float | End-to-end response latency |

## 2) `evaluation_scores_euf_context.xlsx`

Single sheet: `scores`

| Column | Type | Description |
| --- | --- | --- |
| id | integer | Row id from scores DB |
| evaluation_id | integer | Foreign key to `evaluations.id` |
| model_name | string | Model identifier |
| language | string | Language code |
| question_id | string | Per-language question id |
| relevance | float [0,1] | Semantic relevance score |
| factual_accuracy | float [0,1] | NLI/context factual score |
| completeness | float [0,1] | Context coverage/completeness |
| fluency | float [0,1] | Fluency score |
| coherence | float [0,1] | Coherence score |
| prompt_alignment | float [0,1] | Prompt alignment score |
| token_efficiency | float [0,1] | Token efficiency score |
| overall_quality | float [0,1] | Weighted composite score |
| timestamp | datetime/text | Scoring timestamp |

## 3) `evaluation_results_euf_context_by_model.xlsx`

Multiple sheets:
- One sheet per model (sheet names may be truncated by Excel limits)
- `all_results` (full combined dataset)

Per-sheet columns match `evaluation_results_euf_context.xlsx`:

- `id`
- `model_name`
- `language`
- `question_id`
- `run_number`
- `question_text`
- `context`
- `response`
- `timestamp`
- `latency_ms`

## Deprecated/Legacy Fields

The following legacy fields are **not** part of current context exports:
- `run_id`
- `model_url`
- `question_category`
- `time_to_first_token_ms`
- `tokens_generated`
- `tokens_prompt`
- `tokens_per_second`
- `error`

If these are needed again, export logic must be extended in evaluation and scoring scripts.
