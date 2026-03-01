# GPU Efficiency Report

**Generated:** 2026-03-01 16:22:08

## Overview

- Source log: `logs/gpu_metrics.csv`
- Samples: **20931**
- Time range: **2026-02-28 14:28:04.390000 → 2026-02-28 21:10:36.158000**
- Duration: **6.71 hours**

## Core GPU Statistics

- Mean GPU utilization: **78.81%**
- P10 GPU utilization: **0.00%**
- P90 GPU utilization: **100.00%**
- Mean GPU memory used: **133.59 GB**
- P10 GPU memory used: **0.00 GB**
- P90 GPU memory used: **162.62 GB**
- Max GPU memory used: **166.43 GB**
- Max GPU temperature: **52.0°C**

## Metric Glossary (Simple Terms)

- **`gpu_util_mean`**: average GPU usage over time for a model.
  Example: `70%` means the GPU was, on average, 70% busy while that model was active.
- **`gpu_util_p90`**: 90th percentile GPU usage.
  Simple meaning: for 90% of samples, usage was at or below this value; only 10% were higher.
- **`gpu_util_p10`**: 10th percentile GPU usage.
  Simple meaning: 10% of samples were at or below this value; this shows low-end utilization periods.
- **`mem_used_max_gb`**: highest GPU memory observed for a model.
  Use this for capacity planning and OOM risk checks.
- **`mem_used_mean_gb`**: average GPU memory used for a model.
  Helps compare typical memory footprint across models.
- **`mem_used_p90_gb`**: high-memory operating level for a model (not just one-off peak).
- **`temp_max_c`**: highest GPU temperature observed.
- **`avg_latency_ms`**: average end-to-end response latency per model from results DB.
- **`avg_overall_quality`**: average final quality score from scores DB (0 to 1).

### Why Percentiles (P10/P90) Matter

- `mean` can hide spikes and dips.
- `P90` shows whether high utilization/memory is sustained or only occasional.
- `P10` shows lower-tail behavior (idle gaps, stalls, loading transitions).
- Quick rule: if `P90` is very high and `P10` is very low, workload is bursty.

### Example Interpretation

- In this run, global `gpu_util_p90` is **100.00%**, so the GPU was near saturation for most active samples.
- Global `mem_used_max_gb` is **166.43 GB** while `mem_used_mean_gb` is **133.59 GB`.
  This indicates high memory residency during evaluation phases with lower memory during loading/idle.

## Performance Highlights

- Best model overall quality: **devstral-small-2-24b-instruct-2512-b200** (0.816)
- Fastest responses (avg latency): **eurollm-9b-instruct-2512** (2751.7 ms)
- Most GPU memory consumed (max): **devstral-small-2-24b-instruct-2512-b200** (166.43 GB)
- Minimum GPU memory consumed (max): **teuken-7b-instruct-commercial-v0.4** (0.00 GB)

## Phase Breakdown

| phase | samples | pct |
| --- | --- | --- |
| evaluating | 16799 | 80.26 |
| loading_model | 4073 | 19.46 |
| stopping_model | 55 | 0.26 |
| idle | 4 | 0.02 |

## Model-Level GPU Efficiency

| model | duration_min | gpu_util_p10 | gpu_util_mean | gpu_util_p90 | mem_used_p10_gb | mem_used_mean_gb | mem_used_p90_gb | mem_used_max_gb | temp_max_c | avg_latency_ms | avg_overall_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-r1-distill-qwen-32b-b200 | 95.44 | 99.00 | 90.75 | 100.00 | 162.62 | 150.07 | 162.62 | 162.62 | 49 | 14418.2 | 0.784 |
| deepseek-r1-distill-qwen-14b-b200 | 57.28 | 0.00 | 84.72 | 100.00 | 28.47 | 141.47 | 162.33 | 162.33 | 47 | 8130.4 | 0.774 |
| deepseek-r1-distill-qwen-7b | 34.32 | 0.00 | 78.20 | 100.00 | 1.95 | 137.77 | 162.32 | 162.32 | 45 | 4736.8 | 0.775 |
| eurollm-22b-instruct-2512 | 34.26 | 0.00 | 76.22 | 100.00 | 0.00 | 129.43 | 162.44 | 162.44 | 49 | 4358.7 | 0.811 |
| mistral-nemo-instruct-2407-b200 | 28.27 | 0.00 | 74.43 | 100.00 | 0.00 | 126.67 | 162.23 | 162.23 | 46 | 3545.2 | 0.806 |
| mistral-small-3-2-24b-instruct-2506-awq-sym-b200 | 29.13 | 0.00 | 72.93 | 100.00 | 0.00 | 123.16 | 162.35 | 162.35 | 46 | 3562.5 | 0.810 |
| eurollm-9b-instruct-2512 | 22.74 | 0.00 | 71.78 | 100.00 | 0.00 | 123.61 | 162.22 | 162.22 | 45 | 2751.7 | 0.812 |
| devstral-small-2-24b-instruct-2512-b200 | 33.28 | 0.00 | 70.07 | 100.00 | 0.00 | 121.53 | 162.36 | 166.43 | 52 | 3896.2 | 0.816 |
| qwen3-30b-a3b-instruct-awq-b200 | 33.31 | 0.00 | 64.14 | 100.00 | 0.00 | 113.42 | 162.52 | 162.52 | 34 | 3612.6 | 0.800 |
| teuken-7b-instruct-commercial-v0.4 | 1.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 32 | NA | NA |

## Generated Artifacts

- Charts: `insights/gpu_efficiency/charts/`
- Data tables: `insights/gpu_efficiency/data/`
- This report: `insights/gpu_efficiency/GPU_EFFICIENCY_REPORT.md`
