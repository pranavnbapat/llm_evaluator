#!/workspace/llm_evaluator/.venv/bin/python3
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

Environment Variables:
    Loads from .env file in the same directory. See .env.sample for available options.
"""
import os
import sys
import re
import yaml
import json
import time
import signal
import sqlite3
import subprocess
import threading
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from translations.eu_24_languages import get_all_questions


def load_env_file(env_path: Path) -> None:
    """Load environment variables from .env file."""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    # Only set if not already set in environment
                    if key not in os.environ:
                        os.environ[key] = value


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in config values.
    
    Supports formats:
        ${VAR} - Substitutes env var or empty string if not set
        ${VAR:-default} - Substitutes env var or default if not set
    """
    if isinstance(value, str):
        # Pattern to match ${VAR} or ${VAR:-default}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match: re.Match) -> str:
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.environ.get(var_name, default)
            else:
                return os.environ.get(var_expr, '')
        
        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    else:
        return value


def load_config() -> dict:
    """Load configuration from config.yaml with environment variable substitution."""
    config_path = Path(__file__).parent / "config.yaml"
    env_path = Path(__file__).parent / ".env"
    
    # Load .env file first
    load_env_file(env_path)
    
    # Load YAML config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = substitute_env_vars(config)
    
    return config


class VLlmManager:
    """Manages vLLM server lifecycle."""
    
    def __init__(self, config: dict):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port = int(config["vllm"]["port"])
        self.host = config["vllm"]["host"]
        self.api_key = config.get("vllm_api_key", "")
        
    def start(self, model_config: dict) -> bool:
        """Start vLLM with given model."""
        model_path = model_config["local_path"]
        
        # Build command - use venv's vllm
        vllm_path = Path(sys.executable).parent / "vllm"
        cmd = [
            str(vllm_path), "serve", model_path,
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


class GPUMonitor:
    """Logs GPU metrics once per second to a CSV file."""

    def __init__(self, output_path: Path, interval_sec: float = 1.0):
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        header = [
            "timestamp",
            "gpu_index",
            "util_gpu_pct",
            "util_mem_pct",
            "mem_total_mb",
            "mem_used_mb",
            "mem_free_mb",
            "temp_c",
            "power_w",
            "cpu_util_pct",
            "ram_total_mb",
            "ram_used_mb",
            "ram_free_mb",
        ]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists()

        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            prev_cpu_total = None
            prev_cpu_idle = None
            while not self._stop_event.is_set():
                try:
                    cpu_util = self._read_cpu_util(prev_cpu_total, prev_cpu_idle)
                    prev_cpu_total, prev_cpu_idle, cpu_pct = cpu_util
                    ram_total_mb, ram_used_mb, ram_free_mb = self._read_ram_mb()

                    cmd = [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
                        "memory.total,memory.used,memory.free,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits",
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
                    for line in lines:
                        row = [v.strip() for v in line.split(",")]
                        row.extend([f"{cpu_pct:.2f}", ram_total_mb, ram_used_mb, ram_free_mb])
                        writer.writerow(row)
                    f.flush()
                except FileNotFoundError:
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "NA", "nvidia-smi not found"])
                    f.flush()
                    return
                except Exception as e:
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), "NA", f"error: {type(e).__name__}"])
                    f.flush()
                time.sleep(self.interval_sec)

    def _read_cpu_util(self, prev_total, prev_idle):
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return prev_total, prev_idle, 0.0
        vals = [int(v) for v in parts[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        if prev_total is None or prev_idle is None:
            return total, idle, 0.0
        total_delta = total - prev_total
        idle_delta = idle - prev_idle
        if total_delta <= 0:
            return total, idle, 0.0
        util = (1.0 - (idle_delta / total_delta)) * 100.0
        return total, idle, max(0.0, min(100.0, util))

    def _read_ram_mb(self):
        mem_total_kb = 0
        mem_available_kb = 0
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available_kb = int(line.split()[1])
        mem_used_kb = max(0, mem_total_kb - mem_available_kb)
        return (
            int(mem_total_kb / 1024),
            int(mem_used_kb / 1024),
            int(mem_available_kb / 1024),
        )


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
        self.num_runs = int(config["evaluation"]["num_runs"])
    
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
    # Load config with environment variable substitution
    config = load_config()
    
    # Initialize components
    vllm = VLlmManager(config)
    evaluator = Evaluator(vllm, config)
    gpu_monitor = GPUMonitor(evaluator.results_dir / "gpu_metrics.csv")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interrupted! Cleaning up...")
        gpu_monitor.stop()
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

    gpu_monitor.start()
    try:
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
    finally:
        gpu_monitor.stop()
    
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
