"""FastAPI main application."""
import os
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from loguru import logger

# Import local modules
from app.config import settings, parse_model_endpoints, ModelEndpoint
from app.schemas import (
    EvaluationRequest,
    EvaluationResult,
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    AggregateMetrics,
    CrossLanguageConsistency,
    EvaluationReport,
)
from app.models import init_db, get_session_factory, store_evaluation_result
from app.services import ModelManager, ResponseLogger, EvaluationPipeline
from metrics import ResponseEvaluator, generate_evaluation_report
from translations.eu_24_languages import get_all_questions, get_question_metadata, EU_LANGUAGES


# Global state
model_manager = ModelManager()
evaluator = ResponseEvaluator(embedding_model_name=settings.embedding_model)
pipeline: Optional[EvaluationPipeline] = None


def load_questions() -> Dict[str, Dict[str, str]]:
    """Load all questions in all languages."""
    return get_all_questions()


def get_reference_data(question_id: str) -> Dict[str, Any]:
    """Get reference data for a question."""
    metadata = get_question_metadata()
    
    # Map question IDs to metadata keys
    q_map = {
        "Q1_FACTUAL_KNOWLEDGE": "Q1",
        "Q2_LOGICAL_REASONING": "Q2",
        "Q3_INSTRUCTION_FOLLOWING": "Q3",
        "Q4_CULTURAL_NUANCE": "Q4",
        "Q5_SUMMARIZATION_ACCURACY": "Q5",
    }
    
    meta_key = q_map.get(question_id)
    if not meta_key or meta_key not in metadata:
        return {}
    
    meta = metadata[meta_key]
    
    # Build reference data based on question type
    ref_data = {
        "expected_elements": meta.get("expected_elements", []),
        "required_keys": meta.get("required_keys", []),
        "max_sentences": meta.get("max_sentences", 3),
    }
    
    # Add question-specific reference facts
    if question_id == "Q1_FACTUAL_KNOWLEDGE":
        ref_data["reference_facts"] = {
            "capital": "Lisbon",
            "population": "500000",
            "eu_membership": "1986",
            "location": "Iberian Peninsula",
        }
    elif question_id == "Q2_LOGICAL_REASONING":
        ref_data["expected_answer"] = meta.get("expected_answer")
    
    return ref_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Evaluator...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Register model endpoints
    endpoints = parse_model_endpoints()
    for ep in endpoints:
        model_manager.register_model(ep.name, ep.url, ep.api_key)
    
    # Initialize pipeline
    global pipeline
    pipeline = EvaluationPipeline(
        model_manager=model_manager,
        response_logger=ResponseLogger(settings.results_dir),
    )
    
    logger.info(f"Registered {len(endpoints)} model endpoint(s)")
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Evaluator...")


# Create FastAPI app
app = FastAPI(
    title="LLM Evaluator",
    description="Scientific evaluation of LLMs across 24 EU languages",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LLM Evaluator",
        "version": "1.0.0",
        "languages_supported": len(EU_LANGUAGES),
        "questions_available": 5,
    }


@app.get("/languages")
async def get_languages():
    """Get list of supported EU languages."""
    return {
        "count": len(EU_LANGUAGES),
        "languages": [
            {"code": code, "name": info["name"], "native": info["native"]}
            for code, info in EU_LANGUAGES.items()
        ],
    }


@app.get("/questions")
async def get_questions():
    """Get all evaluation questions."""
    metadata = get_question_metadata()
    return {
        "count": len(metadata),
        "questions": metadata,
    }


@app.post("/evaluate", response_model=Dict[str, Any])
async def evaluate(request: EvaluationRequest):
    """Run evaluation for a single model configuration."""
    
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    # Register/override model when a URL is provided
    if request.model_url:
        model_manager.register_model(
            request.model_name,
            request.model_url,
            request.api_key or ""
        )
    
    # Load questions
    all_questions = load_questions()
    
    # Build evaluation configurations
    evaluations = []
    languages = request.languages or ["EN"]
    question_filter = set(request.questions) if request.questions else None
    
    for lang in languages:
        if lang not in all_questions:
            continue
        
        for qid, qtext in all_questions[lang].items():
            if question_filter and qid not in question_filter:
                continue
            
            # Get question category
            metadata = get_question_metadata()
            q_key = qid.replace("Q1_", "Q1").replace("Q2_", "Q2").replace("Q3_", "Q3").replace("Q4_", "Q4").replace("Q5_", "Q5")
            q_key = qid[:2] if qid[:2] in metadata else qid
            category = metadata.get(q_key, {}).get("category", "unknown")
            
            # Add multiple runs for reproducibility
            for run_idx in range(request.num_runs):
                evaluations.append({
                    "model_name": request.model_name,
                    "question_id": qid,
                    "question_text": qtext,
                    "question_category": category,
                    "language": lang,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                })
    
    # Run evaluations
    logger.info(f"Starting evaluation: {len(evaluations)} runs for {request.model_name}")
    results = await pipeline.run_batch(
        evaluations,
        batch_size=settings.batch_size,
        delay=settings.delay_between_requests,
    )
    
    # Evaluate quality scores
    evaluated_results = []
    for result in results:
        if result.error:
            evaluated_results.append(result)
            continue
        
        # Get reference data
        ref_data = get_reference_data(result.question_id)
        
        # Compute quality scores
        scores = evaluator.evaluate_response(
            question_id=result.question_id,
            question_text=result.question_text,
            response_text=result.response_text,
            language=result.language,
            tokens_generated=result.raw_metrics.tokens_generated,
            reference_data=ref_data,
        )
        
        result.quality_scores = scores
        evaluated_results.append(result)
    
    # Store in database
    session_factory = get_session_factory()
    with session_factory() as session:
        for result in evaluated_results:
            store_evaluation_result(session, {
                "run_id": result.run_id,
                "timestamp": result.timestamp,
                "model_name": result.model_name,
                "model_config": result.model_settings,
                "question_id": result.question_id,
                "question_category": result.question_category,
                "language": result.language,
                "question_text": result.question_text,
                "response_text": result.response_text,
                "raw_metrics": result.raw_metrics.model_dump(),
                "quality_scores": result.quality_scores.model_dump() if result.quality_scores else None,
                "error": result.error,
                "metadata": result.metadata,
            })
    
    # Calculate summary
    successful = [r for r in evaluated_results if not r.error]
    avg_score = sum(r.quality_scores.overall_quality for r in successful if r.quality_scores) / len(successful) if successful else 0
    
    return {
        "status": "completed",
        "model": request.model_name,
        "total_runs": len(evaluations),
        "successful": len(successful),
        "failed": len([r for r in evaluated_results if r.error]),
        "average_quality_score": round(avg_score, 4),
        "results": [
            {
                "run_id": r.run_id,
                "question_id": r.question_id,
                "language": r.language,
                "overall_score": r.quality_scores.overall_quality if r.quality_scores else None,
                "latency_ms": r.raw_metrics.latency_ms,
                "error": r.error,
            }
            for r in evaluated_results
        ],
    }


@app.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks,
):
    """Run batch evaluation across multiple models."""
    
    batch_id = str(uuid.uuid4())
    total_runs = sum(
        len(r.languages or ["EN"]) * 5 * r.num_runs
        for r in request.models
    )
    
    # Start background evaluation
    background_tasks.add_task(run_batch_evaluation, batch_id, request)
    
    return BatchEvaluationResponse(
        batch_id=batch_id,
        status="pending",
        total_runs=total_runs,
    )


async def run_batch_evaluation(batch_id: str, request: BatchEvaluationRequest):
    """Background task for batch evaluation."""
    logger.info(f"Starting batch evaluation {batch_id}")
    
    all_results = []
    errors = []
    
    for model_request in request.models:
        try:
            # Evaluate single model
            result = await evaluate(model_request)
            all_results.append(result)
            
            # If running sequentially, wait between models
            if request.run_sequential:
                await asyncio.sleep(5)  # Time for GPU switch
                
        except Exception as e:
            logger.error(f"Error evaluating {model_request.model_name}: {e}")
            errors.append(f"{model_request.model_name}: {str(e)}")
    
    # Save results
    if request.save_intermediate:
        results_file = f"{settings.results_dir}/batch_{batch_id}.json"
        with open(results_file, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "completed_at": datetime.utcnow().isoformat(),
                "results": all_results,
                "errors": errors,
            }, f, indent=2, default=str)


@app.get("/results/{model_name}")
async def get_results(
    model_name: str,
    language: Optional[str] = None,
    question_id: Optional[str] = None,
):
    """Get evaluation results for a model."""
    
    session_factory = get_session_factory()
    with session_factory() as session:
        from app.models import EvaluationRun
        
        query = session.query(EvaluationRun).filter(EvaluationRun.model_name == model_name)
        
        if language:
            query = query.filter(EvaluationRun.language == language)
        if question_id:
            query = query.filter(EvaluationRun.question_id == question_id)
        
        runs = query.all()
        
        return {
            "model": model_name,
            "count": len(runs),
            "results": [
                {
                    "run_id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "question_id": r.question_id,
                    "language": r.language,
                    "overall_score": r.score_overall,
                    "latency_ms": r.latency_ms,
                }
                for r in runs
            ],
        }


@app.get("/report/{model_name}")
async def get_report(model_name: str):
    """Generate comprehensive evaluation report for a model."""
    
    session_factory = get_session_factory()
    with session_factory() as session:
        from app.models import EvaluationRun
        
        runs = session.query(EvaluationRun).filter(
            EvaluationRun.model_name == model_name
        ).all()
        
        if not runs:
            raise HTTPException(status_code=404, detail=f"No results found for {model_name}")
        
        # Convert to DataFrame
        data = []
        for r in runs:
            data.append({
                "model_name": r.model_name,
                "question_id": r.question_id,
                "question_category": r.question_category,
                "language": r.language,
                "score_overall": r.score_overall,
                "score_relevance": r.score_relevance,
                "score_factual_accuracy": r.score_factual_accuracy,
                "score_completeness": r.score_completeness,
                "score_fluency": r.score_fluency,
                "score_coherence": r.score_coherence,
                "latency_ms": r.latency_ms,
                "tokens_per_second": r.tokens_per_second,
            })
        
        df = pd.DataFrame(data)
        report = generate_evaluation_report(df, model_name)
        
        return report


@app.get("/export/{model_name}")
async def export_results(
    model_name: str,
    format: str = Query("json", enum=["json", "csv", "xlsx"]),
):
    """Export evaluation results to file."""
    
    session_factory = get_session_factory()
    with session_factory() as session:
        from app.models import EvaluationRun
        
        runs = session.query(EvaluationRun).filter(
            EvaluationRun.model_name == model_name
        ).all()
        
        if not runs:
            raise HTTPException(status_code=404, detail=f"No results found for {model_name}")
        
        # Prepare export data
        export_data = []
        for r in runs:
            export_data.append({
                "run_id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "model_name": r.model_name,
                "question_id": r.question_id,
                "question_category": r.question_category,
                "language": r.language,
                "question_text": r.question_text,
                "response_text": r.response_text,
                "latency_ms": r.latency_ms,
                "tokens_generated": r.tokens_generated,
                "tokens_per_second": r.tokens_per_second,
                "score_overall": r.score_overall,
                "score_relevance": r.score_relevance,
                "score_factual_accuracy": r.score_factual_accuracy,
                "score_completeness": r.score_completeness,
                "score_fluency": r.score_fluency,
                "score_coherence": r.score_coherence,
                "score_prompt_alignment": r.score_prompt_alignment,
                "score_token_efficiency": r.score_token_efficiency,
                "error": r.error,
            })
        
        df = pd.DataFrame(export_data)
        
        # Export to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{settings.results_dir}/{model_name}_{timestamp}"
        
        if format == "json":
            filepath = f"{filename}.json"
            df.to_json(filepath, orient="records", indent=2, force_ascii=False)
        elif format == "csv":
            filepath = f"{filename}.csv"
            df.to_csv(filepath, index=False, encoding="utf-8")
        elif format == "xlsx":
            filepath = f"{filename}.xlsx"
            df.to_excel(filepath, index=False, engine="openpyxl")
        
        return FileResponse(
            filepath,
            filename=os.path.basename(filepath),
            media_type="application/octet-stream",
        )


@app.get("/compare")
async def compare_models(model_names: List[str] = Query(...)):
    """Compare multiple models statistically."""
    from metrics import paired_ttest, cohens_d
    
    session_factory = get_session_factory()
    with session_factory() as session:
        from app.models import EvaluationRun
        
        # Get results for all models
        model_scores = {}
        for model_name in model_names:
            runs = session.query(EvaluationRun).filter(
                EvaluationRun.model_name == model_name,
                EvaluationRun.score_overall.isnot(None)
            ).all()
            model_scores[model_name] = [r.score_overall for r in runs]
        
        # Pairwise comparisons
        comparisons = []
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                scores_a = model_scores.get(model_a, [])
                scores_b = model_scores.get(model_b, [])
                
                if len(scores_a) == len(scores_b) and len(scores_a) > 1:
                    t_stat, p_value = paired_ttest(scores_a, scores_b)
                    effect = cohens_d(scores_a, scores_b)
                    
                    comparisons.append({
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_a": sum(scores_a) / len(scores_a) if scores_a else 0,
                        "mean_b": sum(scores_b) / len(scores_b) if scores_b else 0,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "cohens_d": effect,
                        "significant": p_value < 0.05,
                    })
        
        return {
            "models": model_names,
            "comparisons": comparisons,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
