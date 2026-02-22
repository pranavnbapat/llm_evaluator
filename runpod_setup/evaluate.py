#!/usr/bin/env python3
"""
Main evaluation script - cycles through all models, runs evaluations, saves results.

Usage:
    export OPENAI_API_KEY=your_key
    python evaluate.py

This script:
    1. For each model in config.yaml:
       a. Starts vLLM with that model
       b. Runs evaluation (asks questions in multiple languages)
       c. Saves results to SQLite and JSON
       d. Stops vLLM
    2. Moves to next model
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

from translations.eu_24_languages import get_all_questions


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
        
        payload = {
            "model": "default",
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
    """Handles evaluation logic."""
    
    def __init__(self, vllm: VLlmManager, config: dict):
        self.vllm = vllm
        self.config = config
        self.results_dir = Path(config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup SQLite
        self.db_path = self.results_dir / "evaluation_results.db"
        self._init_db()
        
        # Load questions
        self.questions = get_all_questions()
        self.languages = config["evaluation"]["languages"]
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
                response TEXT,
                timestamp TEXT,
                latency_ms REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def evaluate_model(self, model_name: str, model_config: dict) -> dict:
        """Evaluate a single model."""
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_config['name']}")
        print(f"{'='*60}")
        
        results = {
            "model_name": model_name,
            "model_display_name": model_config["name"],
            "timestamp": datetime.now().isoformat(),
            "languages": {},
            "total_questions": 0,
            "successful_responses": 0
        }
        
        # Evaluate each language
        for lang in self.languages:
            print(f"\n🌍 Language: {lang}")
            lang_results = []
            
            # Get questions for this language
            lang_questions = self.questions.get(lang, {})
            if not lang_questions:
                print(f"   ⚠️ No questions for {lang}, skipping")
                continue
            
            for qid, question_text in tqdm(lang_questions.items(), desc=f"   {lang}"):
                for run in range(1, self.num_runs + 1):
                    start_time = time.time()
                    
                    response = self.vllm.chat_completion(
                        messages=[{"role": "user", "content": question_text}],
                        temperature=self.config["evaluation"]["temperature"],
                        max_tokens=self.config["evaluation"]["max_tokens"]
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if response:
                        results["successful_responses"] += 1
                        self._save_result(model_name, lang, qid, run, response, latency)
                    
                    results["total_questions"] += 1
            
            results["languages"][lang] = lang_results
        
        # Save JSON summary
        json_path = self.results_dir / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_path}")
        print(f"   Total questions: {results['total_questions']}")
        print(f"   Successful: {results['successful_responses']}")
        
        return results
    
    def _save_result(self, model: str, lang: str, qid: str, run: int, response: str, latency: float):
        """Save result to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evaluations (model_name, language, question_id, run_number, response, timestamp, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model, lang, qid, run, response, datetime.now().isoformat(), latency))
        
        conn.commit()
        conn.close()


def main():
    """Main entry point."""
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check OpenAI key (for evaluation)
    openai_key = os.getenv("OPENAI_API_KEY") or config.get("openai_api_key")
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not set. Evaluation metrics may fail.")
    else:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    # Initialize components
    vllm = VLlmManager(config)
    evaluator = Evaluator(vllm, config)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interrupted! Cleaning up...")
        vllm.stop()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run evaluations
    models = config["models"]
    
    print("="*60)
    print("  LLM Batch Evaluation")
    print("="*60)
    print(f"\nModels to evaluate: {len(models)}")
    print(f"Languages: {', '.join(config['evaluation']['languages'])}")
    print(f"Runs per question: {config['evaluation']['num_runs']}")
    print("")
    
    all_results = []
    
    for model_name, model_config in models.items():
        # Check model exists
        model_path = Path(model_config["local_path"])
        if not model_path.exists():
            print(f"\n❌ Model not found: {model_path}")
            print(f"   Run download_models.py first")
            continue
        
        # Start vLLM
        if not vllm.start(model_config):
            print(f"\n❌ Failed to start vLLM for {model_name}")
            continue
        
        try:
            # Run evaluation
            results = evaluator.evaluate_model(model_name, model_config)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
        finally:
            # Always stop vLLM
            vllm.stop()
        
        # Cool down between models
        time.sleep(10)
    
    # Final summary
    print("\n" + "="*60)
    print("  Evaluation Complete!")
    print("="*60)
    print(f"\nEvaluated {len(all_results)}/{len(models)} models")
    print(f"\nResults saved to:")
    print(f"   SQLite: {evaluator.db_path}")
    print(f"   JSON:   {evaluator.results_dir}/")


if __name__ == "__main__":
    main()
