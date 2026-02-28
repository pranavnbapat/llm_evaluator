"""Evaluation metrics package."""
from .scientific_metrics import ResponseEvaluator, QualityScores
try:
    from .statistical_analysis import (
        compute_summary,
        compute_icc,
        paired_ttest,
        cohens_d,
        anova_one_way,
        bootstrap_confidence_interval,
        cross_language_consistency,
        generate_evaluation_report,
        StatisticalSummary,
    )
    _HAS_STATS = True
except Exception:
    # Keep metrics package importable when optional stats/report deps are not installed.
    _HAS_STATS = False

__all__ = [
    "ResponseEvaluator",
    "QualityScores",
]

if _HAS_STATS:
    __all__.extend([
        "compute_summary",
        "compute_icc",
        "paired_ttest",
        "cohens_d",
        "anova_one_way",
        "bootstrap_confidence_interval",
        "cross_language_consistency",
        "generate_evaluation_report",
        "StatisticalSummary",
    ])
