# Context Evaluation Insights Report

**Generated:** 2026-03-01 15:58:07

## Executive Summary

- Total scored responses: **3240**
- Total evaluation responses (source DB): **3240**
- Coverage (scores / results): **100.00%**
- Models evaluated: **9**
- Languages covered: **24**
- Question families: **5**
- Best model by average overall quality: **devstral-small-2-24b-instruct-2512-b200** (**0.816**)

## Data Integrity Checks

- `evaluation_scores_euf_context.db` rows: **3240**
- `evaluation_scores_euf_context.xlsx` rows: **3240**
- `evaluation_results_euf_context.db` rows: **3240**
- `evaluation_results_euf_context.xlsx` rows: **3240**
- `evaluation_results_euf_context_by_model.xlsx` (`all_results`) rows: **3240**
- Run-count distribution per (model, language, question): **{3: 1080}**

### Score Range Validation

| metric | out_of_range_count | null_count |
| --- | --- | --- |
| relevance | 0 | 0 |
| factual_accuracy | 0 | 0 |
| completeness | 0 | 0 |
| fluency | 0 | 0 |
| coherence | 0 | 0 |
| prompt_alignment | 0 | 0 |
| token_efficiency | 0 | 0 |
| overall_quality | 0 | 0 |

## Model Ranking

| model_name | n | avg_overall | std_overall | p10 | p90 |
| --- | --- | --- | --- | --- | --- |
| devstral-small-2-24b-instruct-2512-b200 | 360 | 0.816 | 0.039 | 0.786 | 0.850 |
| eurollm-9b-instruct-2512 | 360 | 0.812 | 0.048 | 0.777 | 0.855 |
| eurollm-22b-instruct-2512 | 360 | 0.811 | 0.044 | 0.774 | 0.850 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 360 | 0.810 | 0.036 | 0.767 | 0.854 |
| mistral-nemo-instruct-2407-b200 | 360 | 0.806 | 0.050 | 0.768 | 0.849 |
| qwen3-30b-a3b-instruct-awq-b200 | 360 | 0.800 | 0.053 | 0.765 | 0.846 |
| deepseek-r1-distill-qwen-32b-b200 | 360 | 0.784 | 0.049 | 0.743 | 0.832 |
| deepseek-r1-distill-qwen-7b | 360 | 0.775 | 0.048 | 0.718 | 0.823 |
| deepseek-r1-distill-qwen-14b-b200 | 360 | 0.774 | 0.052 | 0.713 | 0.830 |

## Metric Breakdown by Model

| model_name | relevance | factual_accuracy | completeness | fluency | coherence | prompt_alignment | token_efficiency | overall_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| devstral-small-2-24b-instruct-2512-b200 | 0.850 | 0.901 | 0.946 | 0.506 | 0.520 | 0.850 | 0.000 | 0.816 |
| eurollm-9b-instruct-2512 | 0.837 | 0.902 | 0.940 | 0.509 | 0.526 | 0.837 | 0.000 | 0.812 |
| eurollm-22b-instruct-2512 | 0.828 | 0.899 | 0.955 | 0.505 | 0.521 | 0.828 | 0.000 | 0.811 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 0.819 | 0.897 | 0.962 | 0.509 | 0.530 | 0.819 | 0.000 | 0.810 |
| mistral-nemo-instruct-2407-b200 | 0.819 | 0.897 | 0.942 | 0.511 | 0.529 | 0.819 | 0.000 | 0.806 |
| qwen3-30b-a3b-instruct-awq-b200 | 0.811 | 0.893 | 0.934 | 0.508 | 0.526 | 0.811 | 0.000 | 0.800 |
| deepseek-r1-distill-qwen-32b-b200 | 0.704 | 0.892 | 0.967 | 0.559 | 0.551 | 0.704 | 0.000 | 0.784 |
| deepseek-r1-distill-qwen-7b | 0.674 | 0.898 | 0.952 | 0.566 | 0.564 | 0.674 | 0.000 | 0.775 |
| deepseek-r1-distill-qwen-14b-b200 | 0.678 | 0.891 | 0.960 | 0.558 | 0.553 | 0.678 | 0.000 | 0.774 |

## Language Insights

### Top 5 Languages (avg overall quality)

| language | n | avg_overall | std_overall |
| --- | --- | --- | --- |
| ES | 135 | 0.817 | 0.021 |
| SK | 135 | 0.816 | 0.027 |
| SL | 135 | 0.814 | 0.030 |
| RO | 135 | 0.814 | 0.031 |
| EN | 135 | 0.813 | 0.032 |

### Bottom 5 Languages (avg overall quality)

| language | n | avg_overall | std_overall |
| --- | --- | --- | --- |
| GA | 135 | 0.679 | 0.073 |
| MT | 135 | 0.714 | 0.084 |
| HU | 135 | 0.795 | 0.035 |
| IT | 135 | 0.799 | 0.037 |
| DE | 135 | 0.802 | 0.033 |

## Question-Level Insights

| base_qid | n | avg_overall | avg_factual | avg_completeness | avg_fluency |
| --- | --- | --- | --- | --- | --- |
| Q1 | 648 | 0.823 | 0.923 | 0.986 | 0.531 |
| Q4 | 648 | 0.811 | 0.939 | 0.859 | 0.526 |
| Q3 | 648 | 0.799 | 0.910 | 0.969 | 0.523 |
| Q5 | 648 | 0.793 | 0.894 | 0.968 | 0.526 |
| Q2 | 648 | 0.768 | 0.817 | 0.971 | 0.522 |

## Latency by Model (from Results DB)

| model_name | n | avg_latency_ms | p90_latency_ms |
| --- | --- | --- | --- |
| eurollm-9b-instruct-2512 | 360 | 2751.7 | 3342.9 |
| mistral-nemo-instruct-2407-b200 | 360 | 3545.2 | 4659.0 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 360 | 3562.5 | 4438.8 |
| qwen3-30b-a3b-instruct-awq-b200 | 360 | 3612.6 | 8103.1 |
| devstral-small-2-24b-instruct-2512-b200 | 360 | 3896.2 | 5021.7 |
| eurollm-22b-instruct-2512 | 360 | 4358.7 | 5363.5 |
| deepseek-r1-distill-qwen-7b | 360 | 4736.8 | 7332.5 |
| deepseek-r1-distill-qwen-14b-b200 | 360 | 8130.4 | 12402.6 |
| deepseek-r1-distill-qwen-32b-b200 | 360 | 14418.2 | 18933.8 |

## Run-to-Run Stability (within model-language-question)

| model_name | mean | median | max |
| --- | --- | --- | --- |
| eurollm-9b-instruct-2512 | 0.0000 | 0.0000 | 0.0006 |
| deepseek-r1-distill-qwen-14b-b200 | 0.0001 | 0.0000 | 0.0090 |
| mistral-nemo-instruct-2407-b200 | 0.0002 | 0.0000 | 0.0069 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 0.0004 | 0.0000 | 0.0183 |
| eurollm-22b-instruct-2512 | 0.0006 | 0.0000 | 0.0213 |
| deepseek-r1-distill-qwen-32b-b200 | 0.0012 | 0.0000 | 0.0200 |
| deepseek-r1-distill-qwen-7b | 0.0023 | 0.0000 | 0.0404 |
| devstral-small-2-24b-instruct-2512-b200 | 0.0057 | 0.0020 | 0.0615 |
| qwen3-30b-a3b-instruct-awq-b200 | 0.0078 | 0.0044 | 0.1083 |

## Best and Worst Scored Responses (Diagnostic)

### Top 10

| model_name | language | question_id | overall_quality | factual_accuracy | fluency |
| --- | --- | --- | --- | --- | --- |
| mistral-nemo-instruct-2407-b200 | DA | Q4_DA | 0.892 | 0.959 | 0.616 |
| mistral-nemo-instruct-2407-b200 | DA | Q4_DA | 0.892 | 0.959 | 0.616 |
| mistral-nemo-instruct-2407-b200 | DA | Q4_DA | 0.892 | 0.959 | 0.616 |
| deepseek-r1-distill-qwen-7b | FI | Q4_FI | 0.891 | 0.956 | 0.828 |
| eurollm-9b-instruct-2512 | IT | Q4_IT | 0.878 | 0.934 | 0.623 |
| eurollm-9b-instruct-2512 | IT | Q4_IT | 0.878 | 0.934 | 0.623 |
| eurollm-9b-instruct-2512 | IT | Q4_IT | 0.878 | 0.934 | 0.623 |
| qwen3-30b-a3b-instruct-awq-b200 | DE | Q4_DE | 0.877 | 0.961 | 0.603 |
| devstral-small-2-24b-instruct-2512-b200 | LT | Q4_LT | 0.877 | 0.957 | 0.552 |
| qwen3-30b-a3b-instruct-awq-b200 | DA | Q4_DA | 0.871 | 0.944 | 0.542 |

### Bottom 10

| model_name | language | question_id | overall_quality | factual_accuracy | fluency |
| --- | --- | --- | --- | --- | --- |
| qwen3-30b-a3b-instruct-awq-b200 | MT | Q4_MT | 0.494 | 0.723 | 0.570 |
| qwen3-30b-a3b-instruct-awq-b200 | MT | Q4_MT | 0.494 | 0.723 | 0.570 |
| qwen3-30b-a3b-instruct-awq-b200 | GA | Q5_GA | 0.557 | 0.681 | 0.455 |
| mistral-nemo-instruct-2407-b200 | GA | Q2_GA | 0.562 | 0.760 | 0.486 |
| mistral-nemo-instruct-2407-b200 | GA | Q2_GA | 0.562 | 0.760 | 0.486 |
| mistral-nemo-instruct-2407-b200 | GA | Q2_GA | 0.562 | 0.760 | 0.486 |
| eurollm-9b-instruct-2512 | GA | Q1_GA | 0.572 | 0.934 | 0.515 |
| eurollm-9b-instruct-2512 | GA | Q1_GA | 0.572 | 0.934 | 0.515 |
| eurollm-9b-instruct-2512 | GA | Q1_GA | 0.572 | 0.934 | 0.515 |
| devstral-small-2-24b-instruct-2512-b200 | GA | Q1_GA | 0.576 | 0.927 | 0.496 |

## Key Insights

- Performance spread across models is **0.042** (best `devstral-small-2-24b-instruct-2512-b200` vs worst `deepseek-r1-distill-qwen-14b-b200`).
- Language spread is **0.138** between highest and lowest average language scores.
- Completeness is strongly affected by context-coverage behavior and may dominate question-level differences.
- Factual-accuracy values are generally high due to NLI/context matching; inspect low outliers for grounding failures.
- This report is deterministic from artifact files and can be regenerated after each scoring run.

## Requested Comparison Tables

Source files used:
- `results/evaluation_scores_euf_context.xlsx`
- `results/evaluation_results_euf_context.xlsx`
- `results/evaluation_results_euf_context_by_model.xlsx` (`all_results` sheet)

Data consistency check:
- `evaluation_scores_euf_context.xlsx`: **3240** rows
- `evaluation_results_euf_context.xlsx`: **3240** rows
- `evaluation_results_euf_context_by_model.xlsx` (`all_results`): **3240** rows
- Models in scoring file: **9**

### Model Labels Used in Tables

| Label | Model Name |
| --- | --- |
| M1 | deepseek-r1-distill-qwen-14b-b200 |
| M2 | deepseek-r1-distill-qwen-32b-b200 |
| M3 | deepseek-r1-distill-qwen-7b |
| M4 | devstral-small-2-24b-instruct-2512-b200 |
| M5 | eurollm-22b-instruct-2512 |
| M6 | eurollm-9b-instruct-2512 |
| M7 | mistral-nemo-instruct-2407-b200 |
| M8 | mistral-small-3-2-24b-instruct-2506-awq-sym-b200 |
| M9 | qwen3-30b-a3b-instruct-awq-b200 |

### 3.3 Language Performance (All 24 EU Languages, All Evaluated Models)

| Rank | Language | Code | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Spanish | ES | 0.830 | 0.800 | 0.809 | 0.825 | 0.822 | 0.827 | 0.818 | 0.817 | 0.806 | 0.817 |
| 2 | Slovak | SK | 0.811 | 0.804 | 0.795 | 0.831 | 0.818 | 0.829 | 0.809 | 0.825 | 0.820 | 0.816 |
| 3 | Slovenian | SL | 0.797 | 0.802 | 0.779 | 0.823 | 0.824 | 0.834 | 0.816 | 0.825 | 0.829 | 0.814 |
| 4 | Romanian | RO | 0.780 | 0.797 | 0.795 | 0.823 | 0.827 | 0.837 | 0.832 | 0.812 | 0.826 | 0.814 |
| 5 | English | EN | 0.792 | 0.801 | 0.787 | 0.812 | 0.837 | 0.830 | 0.818 | 0.829 | 0.811 | 0.813 |
| 6 | Danish | DA | 0.794 | 0.788 | 0.790 | 0.822 | 0.827 | 0.831 | 0.834 | 0.828 | 0.798 | 0.812 |
| 7 | Swedish | SV | 0.778 | 0.792 | 0.789 | 0.818 | 0.827 | 0.829 | 0.825 | 0.828 | 0.824 | 0.812 |
| 8 | Croatian | HR | 0.787 | 0.801 | 0.771 | 0.833 | 0.819 | 0.821 | 0.820 | 0.836 | 0.819 | 0.812 |
| 9 | Bulgarian | BG | 0.773 | 0.818 | 0.817 | 0.814 | 0.819 | 0.807 | 0.820 | 0.811 | 0.823 | 0.811 |
| 10 | Dutch | NL | 0.807 | 0.807 | 0.784 | 0.825 | 0.824 | 0.824 | 0.821 | 0.805 | 0.792 | 0.810 |
| 11 | French | FR | 0.773 | 0.798 | 0.778 | 0.833 | 0.825 | 0.830 | 0.823 | 0.825 | 0.797 | 0.809 |
| 12 | Czech | CS | 0.793 | 0.806 | 0.796 | 0.828 | 0.818 | 0.824 | 0.805 | 0.806 | 0.798 | 0.808 |
| 13 | Estonian | ET | 0.772 | 0.800 | 0.749 | 0.834 | 0.809 | 0.826 | 0.822 | 0.830 | 0.824 | 0.807 |
| 14 | Finnish | FI | 0.779 | 0.787 | 0.802 | 0.825 | 0.812 | 0.819 | 0.812 | 0.818 | 0.809 | 0.807 |
| 15 | Polish | PL | 0.772 | 0.787 | 0.784 | 0.823 | 0.822 | 0.831 | 0.805 | 0.808 | 0.828 | 0.807 |
| 16 | Latvian | LV | 0.780 | 0.790 | 0.775 | 0.822 | 0.814 | 0.823 | 0.814 | 0.813 | 0.813 | 0.805 |
| 17 | Greek | EL | 0.777 | 0.777 | 0.782 | 0.823 | 0.813 | 0.811 | 0.818 | 0.814 | 0.808 | 0.803 |
| 18 | Portuguese | PT | 0.781 | 0.795 | 0.805 | 0.831 | 0.826 | 0.805 | 0.813 | 0.772 | 0.790 | 0.802 |
| 19 | Lithuanian | LT | 0.764 | 0.781 | 0.758 | 0.834 | 0.812 | 0.820 | 0.812 | 0.804 | 0.834 | 0.802 |
| 20 | German | DE | 0.767 | 0.775 | 0.778 | 0.824 | 0.821 | 0.817 | 0.811 | 0.812 | 0.810 | 0.802 |
| 21 | Italian | IT | 0.771 | 0.784 | 0.790 | 0.816 | 0.818 | 0.816 | 0.812 | 0.800 | 0.785 | 0.799 |
| 22 | Hungarian | HU | 0.776 | 0.783 | 0.744 | 0.820 | 0.806 | 0.793 | 0.816 | 0.804 | 0.815 | 0.795 |
| 23 | Maltese | MT | 0.691 | 0.682 | 0.695 | 0.755 | 0.733 | 0.755 | 0.691 | 0.769 | 0.657 | 0.714 |
| 24 | Irish | GA | 0.637 | 0.651 | 0.646 | 0.697 | 0.692 | 0.655 | 0.688 | 0.750 | 0.693 | 0.679 |

### 3.4 Detailed Metric Breakdown (All Evaluated Models)

| Model | Relevance | Factual | Complete | Fluency | Coherence | Alignment | Efficiency | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.678 | 0.891 | 0.960 | 0.558 | 0.553 | 0.678 | 0.000 | 0.774 |
| M2 | 0.704 | 0.892 | 0.967 | 0.559 | 0.551 | 0.704 | 0.000 | 0.784 |
| M3 | 0.674 | 0.898 | 0.952 | 0.566 | 0.564 | 0.674 | 0.000 | 0.775 |
| M4 | 0.850 | 0.901 | 0.946 | 0.506 | 0.520 | 0.850 | 0.000 | 0.816 |
| M5 | 0.828 | 0.899 | 0.955 | 0.505 | 0.521 | 0.828 | 0.000 | 0.811 |
| M6 | 0.837 | 0.902 | 0.940 | 0.509 | 0.526 | 0.837 | 0.000 | 0.812 |
| M7 | 0.819 | 0.897 | 0.942 | 0.511 | 0.529 | 0.819 | 0.000 | 0.806 |
| M8 | 0.819 | 0.897 | 0.962 | 0.509 | 0.530 | 0.819 | 0.000 | 0.810 |
| M9 | 0.811 | 0.893 | 0.934 | 0.508 | 0.526 | 0.811 | 0.000 | 0.800 |
