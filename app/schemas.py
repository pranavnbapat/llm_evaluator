"""Pydantic schemas for request/response validation."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Categories of evaluation questions."""
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    LOGICAL_REASONING = "logical_reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    CULTURAL_NUANCE = "cultural_nuance"
    SUMMARIZATION = "summarization"


class LanguageCode(str, Enum):
    """ISO 639-1 codes for EU languages."""
    BG = "BG"  # Bulgarian
    HR = "HR"  # Croatian
    CS = "CS"  # Czech
    DA = "DA"  # Danish
    NL = "NL"  # Dutch
    EN = "EN"  # English
    ET = "ET"  # Estonian
    FI = "FI"  # Finnish
    FR = "FR"  # French
    DE = "DE"  # German
    EL = "EL"  # Greek
    HU = "HU"  # Hungarian
    GA = "GA"  # Irish
    IT = "IT"  # Italian
    LV = "LV"  # Latvian
    LT = "LT"  # Lithuanian
    MT = "MT"  # Maltese
    PL = "PL"  # Polish
    PT = "PT"  # Portuguese
    RO = "RO"  # Romanian
    SK = "SK"  # Slovak
    SL = "SL"  # Slovenian
    ES = "ES"  # Spanish
    SV = "SV"  # Swedish


class EvaluationRequest(BaseModel):
    """Request to run evaluation on a model."""
    model_name: str = Field(..., description="Name of the model to evaluate")
    model_url: Optional[str] = Field(None, description="Override URL for the model")
    api_key: Optional[str] = Field(None, description="Override API key")
    languages: List[str] = Field(default=["EN"], description="List of language codes to evaluate")
    questions: Optional[List[str]] = Field(None, description="Specific question IDs to run (None = all)")
    num_runs: int = Field(default=1, ge=1, le=10, description="Number of runs per question for reproducibility")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class RawMetrics(BaseModel):
    """Raw performance metrics from a single evaluation."""
    latency_ms: float = Field(..., description="Total response time in milliseconds")
    time_to_first_token_ms: Optional[float] = Field(None, description="Time to first token")
    tokens_generated: int = Field(..., description="Number of tokens in response")
    tokens_prompt: int = Field(..., description="Number of tokens in prompt")
    tokens_per_second: float = Field(..., description="Generation throughput")


class QualityScores(BaseModel):
    """Computed quality scores for a response."""
    relevance: float = Field(ge=0.0, le=1.0)
    factual_accuracy: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    fluency: float = Field(ge=0.0, le=1.0)
    coherence: float = Field(ge=0.0, le=1.0)
    prompt_alignment: float = Field(ge=0.0, le=1.0)
    token_efficiency: float = Field(ge=0.0, le=1.0)
    overall_quality: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Complete result of a single evaluation run."""
    # Identification
    run_id: str = Field(..., description="Unique identifier for this run")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Model info
    model_name: str
    model_config: Optional[Dict[str, Any]] = None
    
    # Question info
    question_id: str
    question_category: str
    language: str
    question_text: str
    
    # Response
    response_text: str
    
    # Metrics
    raw_metrics: RawMetrics
    quality_scores: Optional[QualityScores] = None
    
    # Additional metadata
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchEvaluationRequest(BaseModel):
    """Request to evaluate multiple models."""
    models: List[EvaluationRequest]
    run_sequential: bool = Field(default=True, description="Run models sequentially to avoid GPU conflicts")
    save_intermediate: bool = Field(default=True)


class BatchEvaluationResponse(BaseModel):
    """Response from batch evaluation."""
    batch_id: str
    status: str  # pending, running, completed, failed
    total_runs: int
    completed_runs: int = 0
    failed_runs: int = 0
    results_file: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class AggregateMetrics(BaseModel):
    """Aggregated metrics across multiple runs."""
    model_name: str
    metric_name: str
    language: Optional[str] = None
    question_id: Optional[str] = None
    
    # Statistics
    n_samples: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p95: float
    ci_lower_95: float
    ci_upper_95: float


class CrossLanguageConsistency(BaseModel):
    """Cross-language consistency metrics."""
    model_name: str
    question_id: str
    
    # Per-language scores
    language_scores: Dict[str, float]
    
    # Consistency metrics
    mean_score: float
    std_score: float
    coefficient_of_variation: float
    cross_language_robustness: float  # mean - 2*std
    
    # Pairwise similarities
    semantic_consistency_matrix: Optional[Dict[str, Dict[str, float]]] = None


class EvaluationReport(BaseModel):
    """Complete evaluation report for a model."""
    report_id: str
    generated_at: datetime
    model_name: str
    
    # Summary statistics
    total_questions: int
    total_languages: int
    total_runs: int
    
    # Scores
    overall_quality_score: float
    cross_language_robustness: float
    
    # Detailed results
    per_language_scores: Dict[str, float]
    per_question_scores: Dict[str, float]
    per_category_scores: Dict[str, float]
    
    # Performance metrics
    avg_latency_ms: float
    avg_tokens_per_second: float
    
    # Consistency metrics
    reproducibility_score: float  # ICC across multiple runs
    temperature_sensitivity: Optional[float] = None
