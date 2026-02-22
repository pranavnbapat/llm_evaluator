# Excel Export Structure

## Sheet 1: Raw Results

| Column | Type | Description |
|--------|------|-------------|
| run_id | string | Unique identifier |
| timestamp | datetime | UTC timestamp |
| model_name | string | Model identifier |
| model_url | string | Endpoint URL |
| question_id | string | Q1-Q5 identifier |
| question_category | string | Category name |
| language | string | ISO 639-1 code |
| question_text | text | Full question |
| response_text | text | Model response |
| latency_ms | float | Response time |
| time_to_first_token_ms | float | TTFT |
| tokens_generated | int | Output tokens |
| tokens_prompt | int | Input tokens |
| tokens_per_second | float | Throughput |
| score_relevance | float [0-1] | Semantic relevance |
| score_factual_accuracy | float [0-1] | Factual correctness |
| score_completeness | float [0-1] | Coverage |
| score_fluency | float [0-1] | Linguistic quality |
| score_coherence | float [0-1] | Logical flow |
| score_prompt_alignment | float [0-1] | Instruction following |
| score_token_efficiency | float [0-1] | Info density |
| score_overall | float [0-1] | Weighted composite |
| error | text | Error message |

## Sheet 2: Aggregate Statistics

| Column | Description |
|--------|-------------|
| model_name | Model identifier |
| metric_name | Name of metric |
| language | Language code (or ALL) |
| question_id | Question (or ALL) |
| n_samples | Number of samples |
| mean | Mean value |
| std | Standard deviation |
| min | Minimum |
| max | Maximum |
| median | Median |
| p95 | 95th percentile |
| ci_lower_95 | Lower 95% CI |
| ci_upper_95 | Upper 95% CI |
| sem | Standard error |

## Sheet 3: Cross-Language Consistency

| Column | Description |
|--------|-------------|
| model_name | Model identifier |
| question_id | Question identifier |
| mean_score | Mean across languages |
| std_score | Standard deviation |
| coefficient_of_variation | CV = std/mean |
| cross_language_robustness | mean - 2*std |
| [Language columns] | Score per language |

## Sheet 4: Model Comparison

| Column | Description |
|--------|-------------|
| model_a | First model |
| model_b | Second model |
| mean_a | Mean score A |
| mean_b | Mean score B |
| difference | Mean difference |
| t_statistic | Paired t-test stat |
| p_value | Significance |
| cohens_d | Effect size |
| significant | p < 0.05 |

## Sheet 5: Configuration

| Parameter | Value |
|-----------|-------|
| Evaluation Date | timestamp |
| Total Models | N |
| Total Languages | 24 |
| Total Questions | 5 |
| Runs per Question | N |
| Temperature | 0.0 |
| Embedding Model | model name |
| Composite Weights | JSON |
