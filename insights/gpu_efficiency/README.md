# GPU Efficiency Insights

This folder contains GPU efficiency analysis for context evaluation runs.

## Output Location

Outputs are generated per run directory:

`results/runs/<gpu_bucket>/<run_id>/insights/gpu_efficiency/`

Contents:

- `GPU_EFFICIENCY_REPORT.md` : main narrative report
- `charts/` : PNG charts
- `data/` : CSV summary tables
- `generate_gpu_efficiency_report.py` : regeneration script (this file)

## Regenerate

From repo root:

Single run:

```bash
python3 insights/gpu_efficiency/generate_gpu_efficiency_report.py --run-dir results/runs/<gpu_bucket>/<run_id>
```

Bulk (all runs under `results/runs/*/*`):

```bash
python3 insights/gpu_efficiency/generate_gpu_efficiency_report.py
```

## Expected Inputs

Per run:

- `results/runs/<gpu_bucket>/<run_id>/logs/gpu_metrics.csv`
- `results/runs/<gpu_bucket>/<run_id>/raw/evaluation_results_euf_context.db`
- `results/runs/<gpu_bucket>/<run_id>/scores/evaluation_scores_euf_context.db`
