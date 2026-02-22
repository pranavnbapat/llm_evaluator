"""Service layer for LLM communication and response handling."""
import time
import uuid
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas import RawMetrics, EvaluationResult


class LLMClient:
    """Client for communicating with vLLM endpoints."""
    
    def __init__(self, base_url: str, api_key: str = "", model: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
    ) -> Dict[str, Any]:
        """Generate a response from the LLM."""
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.default_max_tokens,
            "top_p": top_p if top_p is not None else settings.default_top_p,
        }
        
        # Track timing
        start_time = time.time()
        first_token_time = None
        
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                data = response.json()
                
                # Extract response text
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice:
                        response_text = choice["message"].get("content", "")
                    elif "text" in choice:
                        response_text = choice["text"]
                    else:
                        response_text = str(choice)
                else:
                    response_text = ""
                
                # Get token counts from usage
                usage = data.get("usage", {})
                tokens_prompt = usage.get("prompt_tokens", 0)
                tokens_generated = usage.get("completion_tokens", 0) or len(response_text.split())
                
                # Calculate throughput
                generation_time = latency_ms / 1000  # rough estimate
                tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
                
                return {
                    "response_text": response_text,
                    "raw_metrics": RawMetrics(
                        latency_ms=latency_ms,
                        time_to_first_token_ms=first_token_time,
                        tokens_generated=tokens_generated,
                        tokens_prompt=tokens_prompt,
                        tokens_per_second=tokens_per_second,
                    ),
                    "model": data.get("model", self.model),
                    "finish_reason": choice.get("finish_reason") if "choices" in data else None,
                }
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}")
                raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response from LLM (for real-time monitoring)."""
        # Implementation for streaming if needed
        pass


class ModelManager:
    """Manages model lifecycle for evaluation."""
    
    def __init__(self):
        self.current_model: Optional[str] = None
        self.clients: Dict[str, LLMClient] = {}
    
    def register_model(self, name: str, url: str, api_key: str = ""):
        """Register a model endpoint."""
        self.clients[name] = LLMClient(base_url=url, api_key=api_key, model=name)
        logger.info(f"Registered model: {name} at {url}")
    
    def get_client(self, name: str) -> LLMClient:
        """Get client for a registered model."""
        if name not in self.clients:
            raise ValueError(f"Model {name} not registered")
        return self.clients[name]
    
    async def switch_model(self, name: str) -> bool:
        """Switch to a different model (for GPU management)."""
        if name not in self.clients:
            raise ValueError(f"Model {name} not registered")
        
        # In a real scenario, this might involve:
        # 1. Unloading current model from GPU
        # 2. Loading new model
        # 3. Health check
        
        self.current_model = name
        logger.info(f"Switched to model: {name}")
        return True
    
    async def health_check(self, name: str) -> bool:
        """Check if a model endpoint is healthy."""
        try:
            client = self.get_client(name)
            # Simple test request
            result = await client.generate("Hi", max_tokens=5)
            return True
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return False


class ResponseLogger:
    """Logs responses for analysis and debugging."""
    
    def __init__(self, log_dir: str = "./results"):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)
    
    def log_response(
        self,
        run_id: str,
        model_name: str,
        question_id: str,
        language: str,
        prompt: str,
        response: str,
        metrics: RawMetrics,
        error: Optional[str] = None,
    ):
        """Log a single response to file."""
        import json
        from datetime import datetime
        
        log_entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "question_id": question_id,
            "language": language,
            "prompt": prompt,
            "response": response,
            "metrics": metrics.model_dump() if metrics else None,
            "error": error,
        }
        
        log_file = f"{self.log_dir}/responses_{datetime.now():%Y%m%d}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


class EvaluationPipeline:
    """Main pipeline for running evaluations."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        response_logger: Optional[ResponseLogger] = None,
    ):
        self.model_manager = model_manager
        self.logger = response_logger or ResponseLogger()
        self.results: List[EvaluationResult] = []
    
    async def run_single_evaluation(
        self,
        model_name: str,
        question_id: str,
        question_text: str,
        question_category: str,
        language: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> EvaluationResult:
        """Run a single evaluation."""
        
        run_id = str(uuid.uuid4())
        client = self.model_manager.get_client(model_name)
        
        try:
            # Generate response
            result = await client.generate(
                prompt=question_text,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Log the response
            self.logger.log_response(
                run_id=run_id,
                model_name=model_name,
                question_id=question_id,
                language=language,
                prompt=question_text,
                response=result["response_text"],
                metrics=result["raw_metrics"],
            )
            
            return EvaluationResult(
                run_id=run_id,
                model_name=model_name,
                question_id=question_id,
                question_category=question_category,
                language=language,
                question_text=question_text,
                response_text=result["response_text"],
                raw_metrics=result["raw_metrics"],
                metadata={
                    "finish_reason": result.get("finish_reason"),
                    "model_version": result.get("model"),
                },
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}/{question_id}/{language}: {e}")
            
            self.logger.log_response(
                run_id=run_id,
                model_name=model_name,
                question_id=question_id,
                language=language,
                prompt=question_text,
                response="",
                metrics=None,
                error=str(e),
            )
            
            return EvaluationResult(
                run_id=run_id,
                model_name=model_name,
                question_id=question_id,
                question_category=question_category,
                language=language,
                question_text=question_text,
                response_text="",
                raw_metrics=RawMetrics(
                    latency_ms=0,
                    tokens_generated=0,
                    tokens_prompt=0,
                    tokens_per_second=0,
                ),
                error=str(e),
            )
    
    async def run_batch(
        self,
        evaluations: List[Dict[str, Any]],
        batch_size: int = 5,
        delay: float = 0.5,
    ) -> List[EvaluationResult]:
        """Run evaluations in batches with rate limiting."""
        results = []
        
        for i in range(0, len(evaluations), batch_size):
            batch = evaluations[i:i + batch_size]
            
            # Run batch concurrently
            batch_tasks = [
                self.run_single_evaluation(**eval_config)
                for eval_config in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Delay between batches
            if i + batch_size < len(evaluations):
                await asyncio.sleep(delay)
        
        return results
