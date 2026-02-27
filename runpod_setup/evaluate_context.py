# runpod_setup/evaluate_context.py

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
import re
import sys
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
from typing import Optional, Any
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
        self.port = int(config["vllm"]["port"])
        self.host = config["vllm"]["host"]
        self.client_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        self.api_key = config.get("vllm_api_key", "")
        
    def start(self, model_config: dict) -> bool:
        """Start vLLM with given model."""
        model_path = model_config["local_path"]
        # Safe, consistent log suffix
        model_name = model_config["name"].replace(" ", "_").replace("-", "_").lower()
        
        # Build command - use venv's vllm
        vllm_path = Path(sys.executable).parent / "vllm"
        cmd = [
            str(vllm_path), "serve", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", "1",
            "--dtype", model_config.get("dtype", "auto"),
            "--max-model-len", str(model_config["max_model_len"]),
            "--gpu-memory-utilization", str(model_config["gpu_memory_util"]),
        ]
        # Only add api-key if actually set
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        
        if model_config.get("quant"):
            cmd.extend(["--quantization", model_config["quant"]])
        
        print(f"\n🚀 Starting vLLM with {model_config['name']}...")
        print(f"   Command: {' '.join(cmd[:6])} ... {' '.join(cmd[-6:])}")
        
        # Start process with logging
        env = os.environ.copy()
        env["HF_HOME"] = self.config["paths"]["cache_dir"]
        env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        # Help with CUDA memory fragmentation (use new env var name)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # Remove deprecated key if exists
        
        # Create log file for this model
        log_file = open(f"/tmp/vllm_{model_name}.log", "w")
        self.log_file = log_file
        
        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True
        )
        
        # Wait for ready
        return self._wait_for_ready(model_name)
    
    def _wait_for_ready(self, model_name: str, timeout: int = 1800) -> bool:
        """Wait for vLLM to be ready."""
        url = f"http://{self.client_host}:{self.port}/health"
        
        for i in range(timeout // 5):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ vLLM ready!")
                    mid = self._get_model_name()
                    print(f"   🔎 vLLM model id: {mid}")
                    return True
            except:
                pass
            
            if self.process.poll() is not None:
                print(f"   ❌ vLLM process died")
                print(f"   📄 Check log: /tmp/vllm_{model_name}.log")
                # Show last 20 lines of log
                try:
                    with open(f"/tmp/vllm_{model_name}.log", "r") as f:
                        lines = f.readlines()
                        print(f"   📝 Last error lines:")
                        for line in lines[-20:]:
                            print(f"      {line.strip()}")
                except:
                    pass
                return False
            
            print(f"   ⏳ Waiting for vLLM... ({i*5}s)")
            time.sleep(5)
        
        print(f"   ❌ Timeout waiting for vLLM")
        print(f"   📄 Check full log: /tmp/vllm_{model_name}.log")
        return False
    
    def _get_model_name(self) -> Optional[str]:
        """Get the model name from vLLM."""
        try:
            # Use client_host for requests (0.0.0.0 is a bind address, not a destination)
            url = f"http://{self.client_host}:{self.port}/v1/models"
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
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
            except:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
                except:
                    pass
            self.process = None
            time.sleep(3)  # Cool down
            print(f"   ✅ Stopped")
        # Close log file
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
    
    def chat_completion(self, messages: list, **kwargs) -> Optional[str]:
        """Send chat completion request. If chat fails (often due to chat templates),
        fallback to plain /v1/completions with a single prompt string."""
        model_name = self._get_model_name() or "default"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)
        
        # 1) Try chat endpoint first
        chat_url = f"http://{self.client_host}:{self.port}/v1/chat/completions"
        chat_payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            r = requests.post(chat_url, json=chat_payload, headers=headers, timeout=240)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            
            # Print the real reason once
            print(f"   ❌ Chat API error: HTTP {r.status_code}")
            try:
                print(f"   ❌ Chat Body: {r.json()}")
            except Exception:
                print(f"   ❌ Chat Body (text): {r.text[:1000]}")
        except Exception as e:
            print(f"   ❌ Chat API exception: {type(e).__name__}: {e}")
        
        # 2) Fallback: plain completions (no chat template needed)
        completions_url = f"http://{self.client_host}:{self.port}/v1/completions"
        prompt_text = messages[-1]["content"] if messages else ""
        
        comp_payload = {
            "model": model_name,
            "prompt": prompt_text,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            print(f"   🔄 Trying fallback to /v1/completions...")
            r2 = requests.post(completions_url, json=comp_payload, headers=headers, timeout=240)
            if r2.status_code != 200:
                print(f"   ❌ Completions API error: HTTP {r2.status_code}")
                try:
                    print(f"   ❌ Completions Body: {r2.json()}")
                except Exception:
                    print(f"   ❌ Completions Body (text): {r2.text[:1000]}")
                return None
            
            data2 = r2.json()
            return data2["choices"][0].get("text", "").strip() or None
        except Exception as e:
            print(f"   ❌ Completions API exception: {type(e).__name__}: {e}")
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
    results_dir = Path(config["paths"]["results_dir"])
    gpu_monitor = GPUMonitor(results_dir / "gpu_metrics.csv")
    
    # Get list of models to evaluate
    models = list(config["models"].values())
    
    print(f"\n📋 Models to evaluate: {len(models)}")
    for m in models:
        print(f"   - {m['name']}")
    
    print(f"\n📝 Questions: 5 questions × 24 languages = 120 total")
    print(f"🔄 Runs per question: {config['evaluation']['num_runs']}")
    print(f"💾 Database: evaluation_results_euf_context.db")
    
    all_results = []
    
    try:
        gpu_monitor.start()
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
    finally:
        gpu_monitor.stop()


if __name__ == "__main__":
    main()
