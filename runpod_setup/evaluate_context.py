#!/workspace/llm_evaluator/.venv/bin/python3
"""
Context-based evaluation script - RAG evaluation with search result context.

Usage:
    export OPENAI_API_KEY=your_key
    python evaluate_context.py

This script:
    1. For each model in config.yaml:
       a. Starts vLLM with that model
       b. Runs evaluation with context (asks questions in multiple languages + context)
       c. Saves results to SQLite and JSON
       d. Stops vLLM
    2. Moves to next model

Database: evaluation_results_euf_context.db
Scores: evaluation_scores_euf_context.db
"""
import os
import sys
import yaml
import json
import time
import signal
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from translations.eu_24_languages_euf_context import get_all_questions_with_context


class VLlmManager:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, config: dict):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port = config["vllm"]["port"]
        self.host = config["vllm"]["host"]
        self.api_key = config.get("vllm_api_key", "")
        
    def start(self, model_config: dict) -> bool:
        """Start vLLM with given model."""
        model_path = model_config["local_path"]
        
        # Build command
        cmd = [
            "vllm", "serve", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--tensor-parallel-size", "1",
            "--dtype", "auto",
            "--max-model-len", str(model_config["max_model_len"]),
            "--gpu-memory-utilization", str(model_config["gpu_memory_util"]),
        ]
        
        if model_config.get("quant"):
            cmd.extend(["--quantization", model_config["quant"]])
        
        print(f"\n🚀 Starting vLLM with {model_config['name']}...")
        print(f"   Command: {' '.join(cmd[:5])} ...")
        
        # Start process
        env = os.environ.copy()
        env["HF_HOME"] = self.config["paths"]["cache_dir"]
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env
        )
        
        # Wait for ready
        return self._wait_for_ready()
    
    def _wait_for_ready(self, timeout: int = 120) -> bool:
        """Wait for vLLM to be ready."""
        url = f"http://{self.host}:{self.port}/health"
        
        for i in range(timeout // 5):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ vLLM ready!")
                    return True
            except:
                pass
            
            if self.process.poll() is not None:
                print(f"   ❌ vLLM process died")
                return False
            
            print(f"   ⏳ Waiting for vLLM... ({i*5}s)")
            time.sleep(5)
        
        print(f"   ❌ Timeout waiting for vLLM")
        return False
    
    def _get_model_name(self) -> Optional[str]:
        """Get the model name from vLLM."""
        try:
            url = f"http://{self.host}:{self.port}/v1/models"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    return data["data"][0]["id"]
        except:
            pass
        return None
    
    def stop(self):
        """Stop vLLM."""
        if self.process:
            print(f"\n🛑 Stopping vLLM...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except:
                self.process.kill()
            self.process = None
            time.sleep(3)  # Cool down
            print(f"   ✅ Stopped")
    
    def chat_completion(self, messages: list, **kwargs) -> Optional[str]:
        """Send chat completion request."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        
        # Get actual model name from vLLM
        model_name = self._get_model_name() or "default"
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"   ❌ API error: {e}")
            return None


class Evaluator:
    """Handles context-based evaluation logic."""
    
    def __init__(self, vllm: VLlmManager, config: dict):
        self.vllm = vllm
        self.config = config
        self.results_dir = Path(config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup SQLite with context-specific name
        self.db_path = self.results_dir / "evaluation_results_euf_context.db"
        self._init_db()
        
        # Load questions WITH CONTEXT
        self.questions = get_all_questions_with_context()
        self.num_runs = config["evaluation"]["num_runs"]
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                language TEXT,
                question_id TEXT,
                run_number INTEGER,
                question_text TEXT,
                context TEXT,
                response TEXT,
                timestamp TEXT,
                latency_ms REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _format_context(self, context: list) -> str:
        """Format context entries into a string for the prompt."""
        if not context:
            return ""
        
        context_parts = []
        for i, entry in enumerate(context, 1):
            title = entry.get('title', '')
            description = entry.get('description', '')
            if title and description:
                context_parts.append(f"[{i}] {title}: {description[:300]}...")
            elif title:
                context_parts.append(f"[{i}] {title}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: dict) -> str:
        """Build prompt with context for RAG evaluation."""
        question_text = question['question']
        context = question.get('context', [])
        language = question.get('language', 'EN')
        
        # Format context
        context_str = self._format_context(context)
        
        # Build RAG-style prompt with explicit language instruction
        prompt = f"""You are an expert agriculture advisor. A farmer has asked you a question. Use the provided search results to give a helpful, accurate response.

SEARCH RESULTS (in English):
{context_str}

FARMER'S QUESTION (in {language}):
{question_text}

IMPORTANT INSTRUCTIONS:
1. Answer in the SAME LANGUAGE as the farmer's question ({language})
2. Provide a COMPREHENSIVE but CONCISE answer (2-4 paragraphs)
3. Use SPECIFIC details from the search results
4. Give PRACTICAL, actionable advice that farmers can implement
5. If the search results don't fully answer the question, provide your best expert knowledge

Your response:"""
        
        return prompt
    
    def evaluate_model(self, model_name: str, model_config: dict) -> dict:
        """Evaluate a single model with context."""
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_config['name']}")
        print(f"  Database: evaluation_results_euf_context.db")
        print(f"{'='*60}")
        
        results = {
            "model_name": model_name,
            "model_display_name": model_config["name"],
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "successful_responses": 0
        }
        
        # Evaluate each question (already includes all languages)
        for question_data in tqdm(self.questions, desc="Questions"):
            lang = question_data['language']
            qid = question_data['question_id']
            question_text = question_data['question']
            context = question_data.get('context', [])
            
            for run in range(1, self.num_runs + 1):
                start_time = time.time()
                
                # Build prompt with context
                prompt = self._build_prompt(question_data)
                
                response = self.vllm.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config["evaluation"]["temperature"],
                    max_tokens=self.config["evaluation"]["max_tokens"]
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response:
                    results["successful_responses"] += 1
                    self._save_result(model_name, lang, qid, run, question_text, context, response, latency)
                
                results["total_questions"] += 1
        
        # Save JSON summary
        json_path = self.results_dir / f"{model_name}_context_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results: {results['successful_responses']}/{results['total_questions']} successful")
        print(f"💾 Saved to: {self.db_path}")
        print(f"📄 JSON: {json_path}")
        
        return results
    
    def _save_result(self, model_name: str, language: str, question_id: str, 
                     run_number: int, question_text: str, context: list, 
                     response: str, latency_ms: float):
        """Save result to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert context to JSON string for storage
        context_json = json.dumps(context) if context else ""
        
        cursor.execute("""
            INSERT INTO evaluations 
            (model_name, language, question_id, run_number, question_text, context, response, timestamp, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            language,
            question_id,
            run_number,
            question_text,
            context_json,
            response,
            datetime.now().isoformat(),
            latency_ms
        ))
        
        conn.commit()
        conn.close()


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main entry point."""
    print("="*60)
    print("EU-FarmBook Context-Based Evaluation")
    print("RAG Evaluation with Search Result Context")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Initialize managers
    vllm = VLlmManager(config)
    
    # Get list of models to evaluate
    models = config["models"]
    
    print(f"\n📋 Models to evaluate: {len(models)}")
    for m in models:
        print(f"   - {m['name']}")
    
    print(f"\n📝 Questions: 5 questions × 24 languages = 120 total")
    print(f"🔄 Runs per question: {config['evaluation']['num_runs']}")
    print(f"💾 Database: evaluation_results_euf_context.db")
    
    all_results = []
    
    try:
        for model_config in models:
            model_name = model_config["name"].replace(" ", "_").lower()
            
            # Start vLLM
            if not vllm.start(model_config):
                print(f"❌ Failed to start vLLM for {model_config['name']}")
                continue
            
            try:
                # Run evaluation
                evaluator = Evaluator(vllm, config)
                results = evaluator.evaluate_model(model_name, model_config)
                all_results.append(results)
                
            finally:
                # Always stop vLLM
                vllm.stop()
        
        # Final summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"\n📊 Summary:")
        for r in all_results:
            print(f"   {r['model_display_name']}: {r['successful_responses']}/{r['total_questions']}")
        
        print(f"\n💾 Results saved to: {config['paths']['results_dir']}/evaluation_results_euf_context.db")
        print("\nNext steps:")
        print("   1. Run scoring: python evaluate_results.py")
        print("   2. Check results in results/evaluation_results_euf_context.db")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        vllm.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
