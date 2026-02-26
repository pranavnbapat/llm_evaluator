"""Statistical analysis utilities for evaluation results."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class StatisticalSummary:
    """Statistical summary of a metric."""
    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    p95: float
    ci_lower_95: float
    ci_upper_95: float
    sem: float  # Standard error of mean


def compute_summary(values: List[float]) -> StatisticalSummary:
    """Compute statistical summary for a list of values."""
    if not values:
        return StatisticalSummary(
            n=0, mean=0, std=0, min=0, max=0,
            median=0, q25=0, q75=0, p95=0,
            ci_lower_95=0, ci_upper_95=0, sem=0
        )
    
    arr = np.array(values)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 0 else 0
    
    # 95% confidence interval using t-distribution
    if n > 1:
        t_value = stats.t.ppf(0.975, n - 1)
        ci_lower = mean - t_value * sem
        ci_upper = mean + t_value * sem
    else:
        ci_lower = ci_upper = mean
    
    return StatisticalSummary(
        n=n,
        mean=float(mean),
        std=float(std),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        median=float(np.median(arr)),
        q25=float(np.percentile(arr, 25)),
        q75=float(np.percentile(arr, 75)),
        p95=float(np.percentile(arr, 95)),
        ci_lower_95=float(ci_lower),
        ci_upper_95=float(ci_upper),
        sem=float(sem),
    )


def compute_icc(
    values: List[List[float]],
    model: str = "icc2_1",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute Intraclass Correlation Coefficient (ICC) with bootstrap CI.
    
    Supported models:
        - "icc2_1" / "two_way_random" / "absolute": ICC(2,1)
        - "icc3_1" / "two_way_mixed" / "consistency": ICC(3,1)
    
    Args:
        values: List of lists, where each inner list is measurements for one subject.
                This should be balanced (same number of raters) for strict ICC assumptions.
        model: ICC model type.
        n_bootstrap: Number of bootstrap resamples for CI.
        confidence: Confidence level for CI.
        random_state: Optional random seed for reproducibility.
    
    Returns:
        (icc_value, ci_half_width)
    """
    if not values or len(values) < 2:
        return 0.0, 0.0
    
    # Convert to numpy array with NaN padding, then drop incomplete rows
    max_len = max(len(v) for v in values)
    data = np.full((len(values), max_len), np.nan)
    for i, v in enumerate(values):
        data[i, :len(v)] = v
    
    # ICC assumes balanced data; drop rows with any NaN
    valid_mask = ~np.any(np.isnan(data), axis=1)
    data = data[valid_mask]
    
    n, k = data.shape if data.size else (0, 0)
    if n < 2 or k < 2:
        return 0.0, 0.0
    
    def _icc_2_1(x: np.ndarray) -> float:
        # Two-way random effects, absolute agreement
        n_, k_ = x.shape
        mean_rows = np.mean(x, axis=1)
        mean_cols = np.mean(x, axis=0)
        grand_mean = np.mean(x)
        
        msr = k_ * np.sum((mean_rows - grand_mean) ** 2) / (n_ - 1)
        msc = n_ * np.sum((mean_cols - grand_mean) ** 2) / (k_ - 1)
        mse = (
            np.sum((x - mean_rows[:, None] - mean_cols[None, :] + grand_mean) ** 2)
            / ((n_ - 1) * (k_ - 1))
        )
        
        denom = msr + (k_ - 1) * mse + (k_ * (msc - mse) / n_)
        if denom == 0:
            return 0.0
        return (msr - mse) / denom
    
    def _icc_3_1(x: np.ndarray) -> float:
        # Two-way mixed effects, consistency
        n_, k_ = x.shape
        mean_rows = np.mean(x, axis=1)
        mean_cols = np.mean(x, axis=0)
        grand_mean = np.mean(x)
        
        msr = k_ * np.sum((mean_rows - grand_mean) ** 2) / (n_ - 1)
        mse = (
            np.sum((x - mean_rows[:, None] - mean_cols[None, :] + grand_mean) ** 2)
            / ((n_ - 1) * (k_ - 1))
        )
        
        denom = msr + (k_ - 1) * mse
        if denom == 0:
            return 0.0
        return (msr - mse) / denom
    
    model_key = model.lower()
    if model_key in {"icc3", "icc3_1", "two_way_mixed", "consistency"}:
        icc_func = _icc_3_1
    else:
        icc_func = _icc_2_1
    
    icc_value = float(icc_func(data))
    
    # Bootstrap CI
    if n_bootstrap <= 0 or n < 3:
        return icc_value, 0.0
    
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = data[idx, :]
        boot.append(icc_func(sample))
    
    boot = np.array(boot)
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot, alpha * 100)
    upper = np.percentile(boot, (1 - alpha) * 100)
    
    ci_half_width = float((upper - lower) / 2)
    return icc_value, ci_half_width


def paired_ttest(
    model_a_scores: List[float],
    model_b_scores: List[float],
) -> Tuple[float, float]:
    """
    Perform paired t-test to compare two models.
    
    Returns:
        (t_statistic, p_value)
    """
    if len(model_a_scores) != len(model_b_scores) or len(model_a_scores) < 2:
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
    return float(t_stat), float(p_value)


def cohens_d(
    group1: List[float],
    group2: List[float],
) -> float:
    """
    Compute Cohen's d effect size.
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((mean1 - mean2) / pooled_std)


def anova_one_way(
    groups: List[List[float]],
) -> Tuple[float, float]:
    """
    One-way ANOVA across multiple groups.
    
    Returns:
        (f_statistic, p_value)
    """
    if len(groups) < 2:
        return 0.0, 1.0
    
    f_stat, p_value = stats.f_oneway(*groups)
    return float(f_stat), float(p_value)


def bootstrap_confidence_interval(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for any statistic (mean by default).
    """
    if not values:
        return 0.0, 0.0
    
    arr = np.array(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return float(ci_lower), float(ci_upper)


def cross_language_consistency(
    language_scores: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Compute cross-language consistency metrics.
    
    Args:
        language_scores: Dict mapping language code to list of scores
    
    Returns:
        Dict with consistency metrics
    """
    # Compute mean score per language
    means = {lang: np.mean(scores) for lang, scores in language_scores.items() if scores}
    
    if not means:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,  # Coefficient of variation
            "robustness": 0.0,  # mean - 2*std
            "min_lang": None,
            "max_lang": None,
        }
    
    values = list(means.values())
    mean_score = np.mean(values)
    std_score = np.std(values, ddof=1) if len(values) > 1 else 0
    cv = std_score / mean_score if mean_score > 0 else 0
    robustness = mean_score - 2 * std_score
    
    return {
        "mean": float(mean_score),
        "std": float(std_score),
        "cv": float(cv),
        "robustness": float(robustness),
        "min_lang": min(means, key=means.get),
        "max_lang": max(means, key=means.get),
        "per_language_means": {k: float(v) for k, v in means.items()},
    }


def generate_evaluation_report(
    results_df: pd.DataFrame,
    model_name: str,
) -> Dict[str, any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        results_df: DataFrame with evaluation results
        model_name: Name of the model
    
    Returns:
        Report dictionary
    """
    # Filter for this model
    model_results = results_df[results_df["model_name"] == model_name]
    
    if model_results.empty:
        return {"error": "No results found for model"}
    
    # Overall statistics
    overall_scores = model_results["score_overall"].dropna()
    overall_summary = compute_summary(overall_scores.tolist())
    
    # Per-language statistics
    per_language = {}
    for lang in model_results["language"].unique():
        lang_scores = model_results[model_results["language"] == lang]["score_overall"].dropna()
        if not lang_scores.empty:
            per_language[lang] = compute_summary(lang_scores.tolist()).mean
    
    # Per-question statistics
    per_question = {}
    for qid in model_results["question_id"].unique():
        q_scores = model_results[model_results["question_id"] == qid]["score_overall"].dropna()
        if not q_scores.empty:
            per_question[qid] = compute_summary(q_scores.tolist()).mean
    
    # Per-category statistics
    per_category = {}
    for cat in model_results["question_category"].unique():
        if pd.isna(cat):
            continue
        cat_scores = model_results[model_results["question_category"] == cat]["score_overall"].dropna()
        if not cat_scores.empty:
            per_category[cat] = compute_summary(cat_scores.tolist()).mean
    
    # Cross-language consistency
    language_scores_dict = {}
    for lang in model_results["language"].unique():
        scores = model_results[model_results["language"] == lang]["score_overall"].dropna().tolist()
        if scores:
            language_scores_dict[lang] = scores
    
    consistency = cross_language_consistency(language_scores_dict)
    
    # Performance metrics
    latency_summary = compute_summary(model_results["latency_ms"].dropna().tolist())
    throughput_summary = compute_summary(model_results["tokens_per_second"].dropna().tolist())
    
    # Reproducibility (ICC) - if multiple runs per question
    reproducibility_scores = []
    for qid in model_results["question_id"].unique():
        for lang in model_results["language"].unique():
            runs = model_results[
                (model_results["question_id"] == qid) & 
                (model_results["language"] == lang)
            ]["score_overall"].dropna().tolist()
            if len(runs) > 1:
                reproducibility_scores.append(runs)
    
    icc, icc_ci = compute_icc(reproducibility_scores) if reproducibility_scores else (0.0, 0.0)
    
    return {
        "model_name": model_name,
        "total_runs": len(model_results),
        "overall_quality_score": overall_summary.mean,
        "overall_ci_95": (overall_summary.ci_lower_95, overall_summary.ci_upper_95),
        "cross_language_robustness": consistency["robustness"],
        "cross_language_cv": consistency["cv"],
        "per_language_scores": per_language,
        "per_question_scores": per_question,
        "per_category_scores": per_category,
        "performance": {
            "avg_latency_ms": latency_summary.mean,
            "latency_p95_ms": latency_summary.p95,
            "avg_tokens_per_second": throughput_summary.mean,
        },
        "reproducibility": {
            "icc": icc,
            "icc_ci": icc_ci,
        },
        "quality_breakdown": {
            "relevance": compute_summary(model_results["score_relevance"].dropna().tolist()).mean,
            "factual_accuracy": compute_summary(model_results["score_factual_accuracy"].dropna().tolist()).mean,
            "completeness": compute_summary(model_results["score_completeness"].dropna().tolist()).mean,
            "fluency": compute_summary(model_results["score_fluency"].dropna().tolist()).mean,
            "coherence": compute_summary(model_results["score_coherence"].dropna().tolist()).mean,
        },
    }
