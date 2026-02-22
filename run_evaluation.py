#!/usr/bin/env python3
"""
CLI script for running evaluations.
Usage:
    python run_evaluation.py --model mistral-7b --languages all --runs 3
    python run_evaluation.py --model llama-2-7b --languages EN DE FR --compare-with mistral-7b
"""
import argparse
import asyncio
import json
import sys
from typing import List

import httpx
from loguru import logger

# EU 24 Languages
ALL_LANGUAGES = [
    "BG", "HR", "CS", "DA", "NL", "EN", "ET", "FI",
    "FR", "DE", "EL", "HU", "GA", "IT", "LV", "LT",
    "MT", "PL", "PT", "RO", "SK", "SL", "ES", "SV"
]


async def run_evaluation(
    api_url: str,
    model_name: str,
    model_url: str,
    api_key: str,
    languages: List[str],
    num_runs: int,
    temperature: float,
) -> dict:
    """Run evaluation via API."""
    
    payload = {
        "model_name": model_name,
        "languages": languages,
        "num_runs": num_runs,
        "temperature": temperature,
    }
    
    if model_url:
        payload["model_url"] = model_url
    if api_key:
        payload["api_key"] = api_key
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/evaluate",
            json=payload,
            timeout=3600,  # 1 hour timeout for large evaluations
        )
        response.raise_for_status()
        return response.json()


async def get_report(api_url: str, model_name: str) -> dict:
    """Get evaluation report."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/report/{model_name}")
        response.raise_for_status()
        return response.json()


async def compare_models(api_url: str, model_names: List[str]) -> dict:
    """Compare models statistically."""
    async with httpx.AsyncClient() as client:
        params = [("model_names", name) for name in model_names]
        response = await client.get(f"{api_url}/compare", params=params)
        response.raise_for_status()
        return response.json()


def print_report(report: dict):
    """Pretty print evaluation report."""
    print("\n" + "="*80)
    print(f"EVALUATION REPORT: {report['model_name']}")
    print("="*80)
    
    print(f"\n📊 OVERALL SCORES:")
    print(f"   Overall Quality Score: {report['overall_quality_score']:.4f}")
    print(f"   95% CI: [{report['overall_ci_95'][0]:.4f}, {report['overall_ci_95'][1]:.4f}]")
    print(f"   Cross-Language Robustness: {report['cross_language_robustness']:.4f}")
    print(f"   Coefficient of Variation: {report['cross_language_cv']:.4f}")
    
    print(f"\n🌍 PER-LANGUAGE SCORES (Top 10):")
    sorted_langs = sorted(
        report['per_language_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for lang, score in sorted_langs:
        print(f"   {lang}: {score:.4f}")
    
    print(f"\n❓ PER-QUESTION SCORES:")
    for qid, score in report['per_question_scores'].items():
        print(f"   {qid}: {score:.4f}")
    
    print(f"\n📁 PER-CATEGORY SCORES:")
    for cat, score in report['per_category_scores'].items():
        print(f"   {cat}: {score:.4f}")
    
    print(f"\n⚡ PERFORMANCE METRICS:")
    perf = report['performance']
    print(f"   Avg Latency: {perf['avg_latency_ms']:.2f} ms")
    print(f"   P95 Latency: {perf['latency_p95_ms']:.2f} ms")
    print(f"   Avg Throughput: {perf['avg_tokens_per_second']:.2f} tok/s")
    
    print(f"\n🔄 REPRODUCIBILITY:")
    rep = report['reproducibility']
    print(f"   ICC: {rep['icc']:.4f} (±{rep['icc_ci']:.4f})")
    
    print(f"\n📈 QUALITY BREAKDOWN:")
    qb = report['quality_breakdown']
    print(f"   Relevance: {qb['relevance']:.4f}")
    print(f"   Factual Accuracy: {qb['factual_accuracy']:.4f}")
    print(f"   Completeness: {qb['completeness']:.4f}")
    print(f"   Fluency: {qb['fluency']:.4f}")
    print(f"   Coherence: {qb['coherence']:.4f}")
    
    print("="*80 + "\n")


def print_comparison(comparison: dict):
    """Pretty print model comparison."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    for comp in comparison['comparisons']:
        print(f"\n{comp['model_a']} vs {comp['model_b']}:")
        print(f"   Mean A: {comp['mean_a']:.4f}")
        print(f"   Mean B: {comp['mean_b']:.4f}")
        print(f"   Difference: {abs(comp['mean_a'] - comp['mean_b']):.4f}")
        print(f"   t-statistic: {comp['t_statistic']:.4f}")
        print(f"   p-value: {comp['p_value']:.4f}")
        print(f"   Cohen's d: {comp['cohens_d']:.4f}")
        print(f"   Significant: {'✓ YES' if comp['significant'] else '✗ NO'}")
    
    print("="*80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="LLM Evaluator CLI")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Evaluator API URL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--model-url", default="", help="Override model URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--languages", nargs="+", default=["EN"], help="Languages to evaluate (or 'all')")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per question")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--report", action="store_true", help="Generate report after evaluation")
    parser.add_argument("--compare-with", nargs="+", help="Compare with other models")
    parser.add_argument("--export", choices=["json", "csv", "xlsx"], help="Export results")
    
    args = parser.parse_args()
    
    # Handle 'all' languages
    languages = ALL_LANGUAGES if "all" in args.languages else args.languages
    
    logger.info(f"Starting evaluation for {args.model}")
    logger.info(f"Languages: {len(languages)} total")
    logger.info(f"Runs per question: {args.runs}")
    
    # Run evaluation
    try:
        result = await run_evaluation(
            api_url=args.api_url,
            model_name=args.model,
            model_url=args.model_url,
            api_key=args.api_key,
            languages=languages,
            num_runs=args.runs,
            temperature=args.temperature,
        )
        
        logger.info(f"Evaluation completed: {result['successful']}/{result['total_runs']} successful")
        logger.info(f"Average quality score: {result['average_quality_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    
    # Generate report
    if args.report:
        try:
            report = await get_report(args.api_url, args.model)
            print_report(report)
        except Exception as e:
            logger.error(f"Failed to get report: {e}")
    
    # Compare models
    if args.compare_with:
        try:
            models_to_compare = [args.model] + args.compare_with
            comparison = await compare_models(args.api_url, models_to_compare)
            print_comparison(comparison)
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")


if __name__ == "__main__":
    asyncio.run(main())
