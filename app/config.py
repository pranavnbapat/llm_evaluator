"""Configuration management for LLM Evaluator."""
import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # LLM API Configuration
    llm_url: str = Field(default="http://localhost:8000/v1/chat/completions", alias="LLM_URL")
    model: str = Field(default="mistral-7b-instruct-v0.2", alias="MODEL")
    api_key: str = Field(default="", alias="API_KEY")
    
    # Request Parameters
    default_temperature: float = Field(default=0.0, alias="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=2048, alias="DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(default=0.95, alias="DEFAULT_TOP_P")
    request_timeout: int = Field(default=120, alias="REQUEST_TIMEOUT")
    
    # Evaluation Parameters
    num_runs_per_question: int = Field(default=3, alias="NUM_RUNS_PER_QUESTION")
    batch_size: int = Field(default=5, alias="BATCH_SIZE")
    delay_between_requests: float = Field(default=0.5, alias="DELAY_BETWEEN_REQUESTS")
    
    # Embedding Model
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        alias="EMBEDDING_MODEL"
    )
    
    # Storage
    database_url: str = Field(default="sqlite:///./results/evaluation_results.db", alias="DATABASE_URL")
    results_dir: str = Field(default="./results", alias="RESULTS_DIR")
    export_format: str = Field(default="jsonl", alias="EXPORT_FORMAT")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default="./results/evaluation.log", alias="LOG_FILE")
    
    # Evaluation Weights
    weight_relevance: float = Field(default=0.25, alias="WEIGHT_RELEVANCE")
    weight_factual_accuracy: float = Field(default=0.20, alias="WEIGHT_FACTUAL_ACCURACY")
    weight_completeness: float = Field(default=0.15, alias="WEIGHT_COMPLETENESS")
    weight_fluency: float = Field(default=0.15, alias="WEIGHT_FLUENCY")
    weight_coherence: float = Field(default=0.10, alias="WEIGHT_COHERENCE")
    weight_prompt_alignment: float = Field(default=0.10, alias="WEIGHT_PROMPT_ALIGNMENT")
    weight_token_efficiency: float = Field(default=0.05, alias="WEIGHT_TOKEN_EFFICIENCY")
    
    @property
    def composite_weights(self) -> dict:
        """Return weights as a dictionary for composite scoring."""
        return {
            "relevance": self.weight_relevance,
            "factual_accuracy": self.weight_factual_accuracy,
            "completeness": self.weight_completeness,
            "fluency": self.weight_fluency,
            "coherence": self.weight_coherence,
            "prompt_alignment": self.weight_prompt_alignment,
            "token_efficiency": self.weight_token_efficiency,
        }


# Global settings instance
settings = Settings()


# Model endpoint management for multi-model evaluation
class ModelEndpoint:
    """Represents a single model endpoint configuration."""
    
    def __init__(self, name: str, url: str, api_key: str = ""):
        self.name = name
        self.url = url
        self.api_key = api_key
    
    def to_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "api_key": self.api_key}


def parse_model_endpoints() -> List[ModelEndpoint]:
    """Parse multiple model endpoints from environment variable."""
    endpoints = []
    
    # Check for MODEL_ENDPOINTS variable
    env_endpoints = os.getenv("MODEL_ENDPOINTS", "")
    if env_endpoints:
        for endpoint_str in env_endpoints.split(","):
            parts = endpoint_str.strip().split("|")
            if len(parts) >= 2:
                name = parts[0]
                url = parts[1]
                api_key = parts[2] if len(parts) > 2 else ""
                endpoints.append(ModelEndpoint(name, url, api_key))
    
    # If no endpoints defined, use default from single model config
    if not endpoints:
        endpoints.append(ModelEndpoint(
            name=settings.model,
            url=settings.llm_url,
            api_key=settings.api_key
        ))
    
    return endpoints
