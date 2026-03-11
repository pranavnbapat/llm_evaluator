"""SQLAlchemy models for database storage."""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

Base = declarative_base()


class EvaluationRun(Base):
    """Database model for evaluation runs."""
    __tablename__ = "evaluation_runs"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Model information
    model_name = Column(String, index=True)
    model_url = Column(String)
    model_config = Column(JSON)
    
    # Question information
    question_id = Column(String, index=True)
    question_category = Column(String, index=True)
    language = Column(String, index=True)
    question_text = Column(Text)
    
    # Response
    response_text = Column(Text)
    
    # Raw metrics
    latency_ms = Column(Float)
    time_to_first_token_ms = Column(Float, nullable=True)
    tokens_generated = Column(Integer)
    tokens_prompt = Column(Integer)
    tokens_per_second = Column(Float)
    
    # Quality scores (0-1 scale)
    score_relevance = Column(Float)
    score_factual_accuracy = Column(Float)
    score_completeness = Column(Float)
    score_fluency = Column(Float)
    score_coherence = Column(Float)
    score_prompt_alignment = Column(Float)
    score_token_efficiency = Column(Float)
    score_overall = Column(Float, index=True)
    
    # Error tracking
    error = Column(Text, nullable=True)
    
    # Keep the DB column name as `metadata`, but avoid the reserved ORM attribute.
    extra_metadata = Column("metadata", JSON, default=dict)


class AggregateResult(Base):
    """Pre-computed aggregate statistics."""
    __tablename__ = "aggregate_results"
    
    id = Column(String, primary_key=True)
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    model_name = Column(String, index=True)
    metric_name = Column(String, index=True)
    language = Column(String, nullable=True, index=True)
    question_id = Column(String, nullable=True, index=True)
    
    n_samples = Column(Integer)
    mean = Column(Float)
    std = Column(Float)
    min_val = Column(Float)
    max_val = Column(Float)
    median = Column(Float)
    p95 = Column(Float)
    ci_lower_95 = Column(Float)
    ci_upper_95 = Column(Float)


# Database engine and session
def get_engine():
    """Create database engine."""
    return create_engine(settings.database_url, echo=False)


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None):
    """Get session factory."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def store_evaluation_result(session, result: Dict[str, Any]):
    """Store an evaluation result in the database."""
    db_result = EvaluationRun(
        id=result["run_id"],
        timestamp=result.get("timestamp", datetime.utcnow()),
        model_name=result["model_name"],
        model_url=result.get("model_url"),
        model_config=result.get("model_config"),
        question_id=result["question_id"],
        question_category=result.get("question_category"),
        language=result["language"],
        question_text=result["question_text"],
        response_text=result["response_text"],
        latency_ms=result["raw_metrics"]["latency_ms"],
        time_to_first_token_ms=result["raw_metrics"].get("time_to_first_token_ms"),
        tokens_generated=result["raw_metrics"]["tokens_generated"],
        tokens_prompt=result["raw_metrics"]["tokens_prompt"],
        tokens_per_second=result["raw_metrics"]["tokens_per_second"],
        score_relevance=result.get("quality_scores", {}).get("relevance"),
        score_factual_accuracy=result.get("quality_scores", {}).get("factual_accuracy"),
        score_completeness=result.get("quality_scores", {}).get("completeness"),
        score_fluency=result.get("quality_scores", {}).get("fluency"),
        score_coherence=result.get("quality_scores", {}).get("coherence"),
        score_prompt_alignment=result.get("quality_scores", {}).get("prompt_alignment"),
        score_token_efficiency=result.get("quality_scores", {}).get("token_efficiency"),
        score_overall=result.get("quality_scores", {}).get("overall_quality"),
        error=result.get("error"),
        extra_metadata=result.get("metadata", {}),
    )
    session.add(db_result)
    session.commit()
    return db_result
