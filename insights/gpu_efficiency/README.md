# GPU Efficiency Insights

This folder contains GPU efficiency analysis for context evaluation runs.

## Outputs

- `GPU_EFFICIENCY_REPORT.md` : main narrative report
- `charts/` : PNG charts
- `data/` : CSV summary tables
- `generate_gpu_efficiency_report.py` : regeneration script

## Regenerate

From repo root:

```bash
python3 insights/gpu_efficiency/generate_gpu_efficiency_report.py
```

## Expected Inputs

- `logs/gpu_metrics.csv`
- `results/evaluation_results_euf_context.db`
- `results/evaluation_scores_euf_context.db`
