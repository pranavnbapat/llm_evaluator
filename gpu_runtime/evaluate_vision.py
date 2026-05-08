# gpu_runtime/evaluate_vision.py

#!/workspace/llm_evaluator/.venv/bin/python3
"""
Vision evaluation script - VLM benchmarking with image+question prompts.

Forks evaluate_context.py: VLlmManager / GPUMonitor / DB scaffolding are kept,
but the prompt is multimodal (text + image_url) and the dataset is loaded from
data/evaluation_vision_questions.json.

Per-model vLLM CLI flags (e.g. --limit-mm-per-prompt) can be supplied via the
optional `vllm_extra_args` list inside each model entry in config.yaml.

Database: evaluation_results_euf_vision.db
"""
import os
import re
import sys
import yaml
import json
import time
import base64
import mimetypes
import signal
import sqlite3
import subprocess
import threading
import csv
import socket
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Union
import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def detect_gpu_bucket() -> tuple[str, str]:
    override = os.getenv("EVAL_RUN_GPU", "").strip().lower()
    if override:
        safe = re.sub(r"[^a-z0-9_\\-]+", "_", override).strip("_")
        return safe or "unknown_gpu", "env:EVAL_RUN_GPU"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        names = [ln.strip().lower() for ln in result.stdout.splitlines() if ln.strip()]
        name = names[0] if names else ""
        if "b200" in name or "gb200" in name:
            return "b200", f"nvidia-smi:{name}"
        if "h200" in name:
            return "h200_sxm", f"nvidia-smi:{name}"
        if "h100" in name:
            return "h100_sxm", f"nvidia-smi:{name}"
        if "l40s" in name or "l40" in name:
            return "l40s", f"nvidia-smi:{name}"
        if "3090" in name:
            return "3090", f"nvidia-smi:{name}"
        if "a100" in name:
            return "a100", f"nvidia-smi:{name}"
        if "a40" in name:
            return "a40", f"nvidia-smi:{name}"
        safe = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return (safe or "unknown_gpu"), f"nvidia-smi:{name or 'unknown'}"
    except Exception:
        return "unknown_gpu", "fallback:unknown"


def resolve_run_paths(base_results_dir: Path) -> dict:
    run_dir_override = os.getenv("EVAL_RUN_DIR", "").strip()
    gpu_bucket, gpu_source = detect_gpu_bucket()
    run_id = os.getenv("EVAL_RUN_ID", "").strip()
    if not run_id:
        run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_vision_eval"
    if run_dir_override:
        run_dir = Path(run_dir_override).expanduser().resolve()
        run_id = run_dir.name
        gpu_bucket = run_dir.parent.name if run_dir.parent.name else gpu_bucket
        run_source = "env:EVAL_RUN_DIR"
    else:
        run_dir = (base_results_dir / "runs" / gpu_bucket / run_id).resolve()
        run_source = "auto"

    raw_dir = run_dir / "raw"
    scores_dir = run_dir / "scores"
    logs_dir = run_dir / "logs"
    insights_dir = run_dir / "insights"
    metadata_dir = run_dir / "metadata"
    for p in [raw_dir, scores_dir, logs_dir, insights_dir, metadata_dir]:
        p.mkdir(parents=True, exist_ok=True)

    latest_root = (base_results_dir / "latest").resolve()
    latest_root.mkdir(parents=True, exist_ok=True)
    latest_link = latest_root / gpu_bucket
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        rel_target = os.path.relpath(run_dir, latest_root)
        latest_link.symlink_to(rel_target)
    except Exception:
        pass

    return {
        "base_results_dir": base_results_dir.resolve(),
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "scores_dir": scores_dir,
        "logs_dir": logs_dir,
        "insights_dir": insights_dir,
        "metadata_dir": metadata_dir,
        "run_id": run_id,
        "gpu_bucket": gpu_bucket,
        "gpu_source": gpu_source,
        "run_source": run_source,
    }


class VLlmManager:
    """Manages vLLM server lifecycle. Supports per-model `vllm_extra_args`."""

    def __init__(self, config: dict):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port = int(config["vllm"]["port"])
        self.host = config["vllm"]["host"]
        self.client_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        self.api_key = config.get("vllm_api_key", "")

    def _is_port_open(self) -> bool:
        try:
            with socket.create_connection((self.client_host, self.port), timeout=1):
                return True
        except Exception:
            return False

    def _kill_stale_vllm(self) -> None:
        if self._is_port_open():
            print(f"   ⚠️ Detected existing server on port {self.port}; cleaning stale vLLM...")
        try:
            subprocess.run(["pkill", "-f", r"\bvllm\b.*\bserve\b"], check=False)
        except Exception:
            pass
        time.sleep(2)

    def start(self, model_config: dict) -> bool:
        self._kill_stale_vllm()
        model_path = model_config["local_path"]
        model_name = model_config["name"].replace(" ", "_").replace("-", "_").lower()

        vllm_path = Path(sys.executable).parent / "vllm"
        cmd = [
            str(vllm_path), "serve", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", "1",
            "--dtype", model_config.get("dtype", "auto"),
            "--max-model-len", str(model_config["max_model_len"]),
            "--gpu-memory-utilization", str(model_config["gpu_memory_util"]),
            "--served-model-name", model_config["name"],
        ]
        if model_config.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        if model_config.get("quant"):
            cmd.extend(["--quantization", model_config["quant"]])

        # Per-model multimodal flags (e.g. --limit-mm-per-prompt image=1).
        extra_args = model_config.get("vllm_extra_args") or []
        for arg in extra_args:
            cmd.append(str(arg))

        print(f"\n🚀 Starting vLLM with {model_config['name']}...")
        print(f"   Command: {' '.join(cmd[:6])} ... {' '.join(cmd[-6:])}")

        env = os.environ.copy()
        env["HF_HOME"] = self.config["paths"]["cache_dir"]
        env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        log_file = open(f"/tmp/vllm_{model_name}.log", "w")
        self.log_file = log_file

        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True
        )
        return self._wait_for_ready(model_name, expected_model_id=model_config["name"])

    def _wait_for_ready(self, model_name: str, expected_model_id: Optional[str] = None, timeout: int = 1800) -> bool:
        url = f"http://{self.client_host}:{self.port}/health"
        for i in range(timeout // 5):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    mid = self._get_model_name()
                    if expected_model_id and mid and mid != expected_model_id:
                        print(f"   ⚠️ Stale server detected (expected '{expected_model_id}', got '{mid}').")
                        self._kill_stale_vllm()
                        return False
                    print(f"   ✅ vLLM ready!")
                    print(f"   🔎 vLLM model id: {mid}")
                    return True
            except:
                pass
            if self.process.poll() is not None:
                print(f"   ❌ vLLM process died")
                print(f"   📄 Check log: /tmp/vllm_{model_name}.log")
                try:
                    with open(f"/tmp/vllm_{model_name}.log", "r") as f:
                        for line in f.readlines()[-80:]:
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
        try:
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
            time.sleep(3)
            print(f"   ✅ Stopped")
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def chat_completion_multimodal(self, content_parts: list, **kwargs) -> Optional[str]:
        """Send a multimodal chat completion. `content_parts` is the OpenAI-style
        list of {type: text|image_url, ...} parts. No /v1/completions fallback —
        plain completions cannot accept image inputs."""
        model_name = self._get_model_name() or "default"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", 2048)

        chat_url = f"http://{self.client_host}:{self.port}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            r = requests.post(chat_url, json=payload, headers=headers, timeout=300)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            print(f"   ❌ Chat API error: HTTP {r.status_code}")
            try:
                print(f"   ❌ Body: {r.json()}")
            except Exception:
                print(f"   ❌ Body (text): {r.text[:1000]}")
        except Exception as e:
            print(f"   ❌ Chat API exception: {type(e).__name__}: {e}")
        return None


class GPUMonitor:
    """Logs GPU metrics once per second to a CSV file."""

    def __init__(self, output_path: Path, interval_sec: float = 1.0):
        self.output_path = output_path
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ctx_lock = threading.Lock()
        self._context = {
            "phase": "idle",
            "model_name": "",
            "model_repo": "",
            "model_dtype": "",
            "model_quant": "",
            "model_max_model_len": "",
            "model_gpu_memory_util": "",
            "eval_language": "",
            "eval_question_id": "",
            "eval_run_number": "",
            "eval_image_ref": "",
            "eval_image_count": "",
            "eval_input_mode": "",
            "eval_max_tokens": "",
            "eval_temperature": "",
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_context(self, **kwargs) -> None:
        with self._ctx_lock:
            for key, value in kwargs.items():
                if key in self._context:
                    self._context[key] = "" if value is None else str(value)

    def _snapshot_context(self) -> dict:
        with self._ctx_lock:
            return dict(self._context)

    def _run(self) -> None:
        header = [
            "timestamp", "gpu_index", "util_gpu_pct", "util_mem_pct",
            "mem_total_mb", "mem_used_mb", "mem_free_mb", "temp_c", "power_w",
            "cpu_util_pct", "ram_total_mb", "ram_used_mb", "ram_free_mb",
            "phase", "model_name", "model_repo", "model_dtype", "model_quant",
            "model_max_model_len", "model_gpu_memory_util",
            "eval_language", "eval_question_id", "eval_run_number",
            "eval_image_ref", "eval_image_count", "eval_input_mode",
            "eval_max_tokens", "eval_temperature",
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
                    ctx = self._snapshot_context()
                    for line in lines:
                        row = [v.strip() for v in line.split(",")]
                        row.extend([f"{cpu_pct:.2f}", ram_total_mb, ram_used_mb, ram_free_mb])
                        row.extend([
                            ctx["phase"], ctx["model_name"], ctx["model_repo"],
                            ctx["model_dtype"], ctx["model_quant"],
                            ctx["model_max_model_len"], ctx["model_gpu_memory_util"],
                            ctx["eval_language"], ctx["eval_question_id"], ctx["eval_run_number"],
                            ctx["eval_image_ref"], ctx["eval_image_count"], ctx["eval_input_mode"],
                            ctx["eval_max_tokens"], ctx["eval_temperature"],
                        ])
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


def load_vision_dataset(dataset_path: Path) -> tuple[list, Path]:
    """Load the vision dataset JSON. Returns (items, image_root_abs_path)."""
    with open(dataset_path) as f:
        data = json.load(f)
    items = data.get("items", [])
    image_root_raw = data.get("image_root", "data/vision_images")
    image_root = Path(image_root_raw)
    if not image_root.is_absolute():
        image_root = (REPO_ROOT / image_root).resolve()
    image_root.mkdir(parents=True, exist_ok=True)
    return items, image_root


def encode_image_as_data_url(path: Path) -> Optional[str]:
    """Read a local image and return a data: URL with detected MIME type."""
    if not path.exists():
        return None
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as e:
        print(f"   ❌ Failed to read image {path}: {type(e).__name__}: {e}")
        return None
    return f"data:{mime};base64,{b64}"


def resolve_image_ref(item: dict, image_root: Path) -> tuple[Optional[str], str]:
    """Resolve item -> (data_url_or_http_url, displayable_ref)."""
    ref = item.get("image_url") or item.get("image_filename") or item.get("image_path")
    if not ref:
        return None, ""
    if isinstance(ref, str) and ref.startswith(("http://", "https://", "data:")):
        return ref, ref
    p = Path(ref)
    if not p.is_absolute():
        p = (image_root / p).resolve()
    return encode_image_as_data_url(p), str(p)


class Evaluator:
    """Vision evaluation: text + image multimodal prompt."""

    def __init__(self, vllm: VLlmManager, config: dict, items: list, image_root: Path,
                 gpu_monitor: Optional[GPUMonitor] = None):
        self.vllm = vllm
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.items = items
        self.image_root = image_root
        self.results_dir = Path(config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.results_dir / "evaluation_results_euf_vision.db"
        self._conn = self._open_db_connection()
        self._init_db()

        self.num_runs = int(config["evaluation"]["num_runs"])

    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                language TEXT,
                question_id TEXT,
                item_id TEXT,
                run_number INTEGER,
                question_text TEXT,
                context TEXT,
                image_ref TEXT,
                image_count INTEGER,
                response TEXT,
                timestamp TEXT,
                latency_ms REAL
            )
        """)
        self._conn.commit()

    def _open_db_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _reconnect_db(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = self._open_db_connection()

    def close(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass

    @staticmethod
    def _format_context(context: list) -> str:
        if not context:
            return ""
        parts = []
        for i, entry in enumerate(context, 1):
            title = entry.get("title", "")
            description = entry.get("description", "")
            if title and description:
                parts.append(f"[{i}] {title}: {description[:300]}...")
            elif title:
                parts.append(f"[{i}] {title}")
        return "\n\n".join(parts)

    @staticmethod
    def _compose_text(question_text: str, language: str, context_str: str) -> str:
        context_block = f"SEARCH RESULTS (in English):\n{context_str}\n\n" if context_str else ""
        return f"""You are an expert agriculture advisor. The farmer has shared an image and a question. Use what you can see in the image (and the optional search results) to give a helpful, accurate response.

{context_block}FARMER'S QUESTION (in {language}):
{question_text}

INSTRUCTIONS:
1. Answer in the SAME LANGUAGE as the question ({language}).
2. Reference what is visible in the image when relevant.
3. Provide PRACTICAL, actionable advice.
4. Be COMPREHENSIVE but CONCISE (2-4 paragraphs).

Your response:"""

    def evaluate_model(self, model_name: str, model_config: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Evaluating (vision): {model_config['name']}")
        print(f"  Database: evaluation_results_euf_vision.db")
        print(f"{'='*60}")

        results = {
            "model_name": model_name,
            "model_display_name": model_config["name"],
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0,
            "successful_responses": 0,
            "skipped_missing_image": 0,
        }

        for item in tqdm(self.items, desc="Items"):
            lang = item.get("language", "EN")
            qid = item.get("question_id", "")
            item_id = item.get("item_id", qid)
            question_text = item.get("question", "")
            context = item.get("context", []) or []

            image_data_url, image_ref_display = resolve_image_ref(item, self.image_root)
            if image_data_url is None:
                print(f"   ⚠️ Skipping {item_id}: image not found ({image_ref_display or 'no image_filename'})")
                results["skipped_missing_image"] += 1
                continue

            text_part = self._compose_text(
                question_text=question_text,
                language=lang,
                context_str=self._format_context(context),
            )
            content_parts = [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": text_part},
            ]

            for run in range(1, self.num_runs + 1):
                if self.gpu_monitor:
                    self.gpu_monitor.set_context(
                        phase="evaluating",
                        eval_language=lang,
                        eval_question_id=qid,
                        eval_run_number=run,
                        eval_image_ref=image_ref_display,
                        eval_image_count=1,
                        eval_input_mode="image+text",
                    )
                start_time = time.time()
                response = self.vllm.chat_completion_multimodal(
                    content_parts=content_parts,
                    temperature=self.config["evaluation"]["temperature"],
                    max_tokens=self.config["evaluation"]["max_tokens"],
                )
                latency = (time.time() - start_time) * 1000

                if response:
                    results["successful_responses"] += 1
                    self._save_result(
                        model_name=model_name,
                        language=lang,
                        question_id=qid,
                        item_id=item_id,
                        run_number=run,
                        question_text=question_text,
                        context=context,
                        image_ref=image_ref_display,
                        image_count=1,
                        response=response,
                        latency_ms=latency,
                    )
                results["total_questions"] += 1

        if self.gpu_monitor:
            self.gpu_monitor.set_context(
                eval_language="", eval_question_id="", eval_run_number="",
                eval_image_ref="", eval_image_count="", eval_input_mode="",
            )

        json_path = self.results_dir / f"{model_name}_vision_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results: {results['successful_responses']}/{results['total_questions']} successful")
        if results["skipped_missing_image"]:
            print(f"🖼️  Skipped (missing image): {results['skipped_missing_image']}")
        print(f"💾 Saved to: {self.db_path}")
        print(f"📄 JSON: {json_path}")
        return results

    def _save_result(self, *, model_name, language, question_id, item_id, run_number,
                     question_text, context, image_ref, image_count, response, latency_ms):
        context_json = json.dumps(context) if context else ""
        for attempt in range(3):
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn.execute("""
                    INSERT INTO evaluations
                    (model_name, language, question_id, item_id, run_number, question_text, context,
                     image_ref, image_count, response, timestamp, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, language, question_id, item_id, run_number,
                    question_text, context_json, image_ref, image_count,
                    response, datetime.now().isoformat(), latency_ms,
                ))
                self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "unable to open database file" in str(e).lower() and attempt < 2:
                    print(f"⚠️ SQLite open failure ({self.db_path}), retrying {attempt + 1}/2...")
                    time.sleep(1.0)
                    self._reconnect_db()
                    continue
                raise


def load_env_file(env_path: Path) -> None:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    if key not in os.environ:
                        os.environ[key] = value


def substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match: re.Match) -> str:
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, "")

        return re.sub(pattern, replace_var, value)
    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    return value


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    env_path = Path(__file__).parent / ".env"
    load_env_file(env_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return substitute_env_vars(config)


def export_results_to_excel(db_path: Path) -> Optional[Path]:
    if not db_path.exists():
        print(f"⚠️ DB not found, skipping Excel export: {db_path}")
        return None
    excel_path = db_path.with_suffix(".xlsx")
    by_model_excel_path = db_path.with_name(f"{db_path.stem}_by_model.xlsx")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY id", conn)
        if df.empty:
            print("⚠️ No evaluation rows found, skipping Excel export.")
            return None
        df.to_excel(excel_path, index=False)
        if "model_name" in df.columns:
            used_sheet_names = set()
            with pd.ExcelWriter(by_model_excel_path, engine="xlsxwriter") as writer:
                for model_name, model_df in df.groupby("model_name", sort=True):
                    base = (str(model_name) if model_name else "unknown_model")
                    safe = "".join("_" if ch in "[]:*?/\\" else ch for ch in base).strip() or "unknown_model"
                    safe = safe[:31]
                    sheet_name = safe
                    n = 1
                    while sheet_name in used_sheet_names:
                        suffix = f"_{n}"
                        sheet_name = f"{safe[:31-len(suffix)]}{suffix}"
                        n += 1
                    used_sheet_names.add(sheet_name)
                    model_df.to_excel(writer, sheet_name=sheet_name, index=False)
                df.to_excel(writer, sheet_name="all_results", index=False)
        return excel_path
    finally:
        conn.close()


def main():
    print("="*60)
    print("EU-FarmBook Vision Evaluation")
    print("VLM Benchmarking with Image+Question Prompts")
    print("="*60)

    config = load_config()
    base_results_dir = Path(config["paths"]["results_dir"]).resolve()
    try:
        run_paths = resolve_run_paths(base_results_dir)
    except Exception as e:
        fallback_base = (Path(__file__).resolve().parent.parent / "results").resolve()
        print(f"⚠️ Could not create run path under '{base_results_dir}': {e}")
        print(f"   Falling back to local results dir: {fallback_base}")
        run_paths = resolve_run_paths(fallback_base)
    config["paths"]["results_dir"] = str(run_paths["raw_dir"])

    dataset_path = Path(os.getenv("EVAL_VISION_DATASET", "")).expanduser()
    if not dataset_path or not str(dataset_path):
        dataset_path = REPO_ROOT / "data" / "evaluation_vision_questions.json"
    if not dataset_path.exists():
        print(f"❌ Vision dataset not found: {dataset_path}")
        sys.exit(2)
    items, image_root = load_vision_dataset(dataset_path)
    print(f"📚 Dataset: {dataset_path}  ({len(items)} items, image_root={image_root})")

    run_meta = {
        "created_at": datetime.now().isoformat(),
        "run_id": run_paths["run_id"],
        "run_dir": str(run_paths["run_dir"]),
        "raw_dir": str(run_paths["raw_dir"]),
        "scores_dir": str(run_paths["scores_dir"]),
        "logs_dir": str(run_paths["logs_dir"]),
        "gpu_bucket": run_paths["gpu_bucket"],
        "gpu_detected_from": run_paths["gpu_source"],
        "run_path_source": run_paths["run_source"],
        "dataset_path": str(dataset_path),
        "image_root": str(image_root),
        "evaluation_mode": "vision",
    }
    with open(run_paths["metadata_dir"] / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    vllm = VLlmManager(config)
    gpu_monitor = GPUMonitor(run_paths["logs_dir"] / "gpu_metrics.csv")

    models = list(config["models"].values())
    print(f"\n📋 Models to evaluate: {len(models)}")
    for m in models:
        extra = m.get("vllm_extra_args") or []
        suffix = f"  (+{len(extra)} extra args)" if extra else ""
        print(f"   - {m['name']}{suffix}")
    print(f"\n🖼️  Items: {len(items)}  ×  runs: {config['evaluation']['num_runs']}")
    print(f"🧭 Run ID: {run_paths['run_id']}")
    print(f"🖥️  GPU bucket: {run_paths['gpu_bucket']} ({run_paths['gpu_source']})")
    print(f"📁 Run dir: {run_paths['run_dir']}")
    print(f"💾 Database: {run_paths['raw_dir'] / 'evaluation_results_euf_vision.db'}")

    all_results = []
    model_statuses = []

    try:
        gpu_monitor.start()
        for model_config in models:
            model_name = model_config["name"].replace(" ", "_").lower()
            model_status = {
                "model_name": model_config.get("name", model_name),
                "repo": model_config.get("repo", ""),
                "status": "pending",
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "details": "",
            }
            gpu_monitor.set_context(
                phase="loading_model",
                model_name=model_config.get("name", model_name),
                model_repo=model_config.get("repo", ""),
                model_dtype=model_config.get("dtype", "auto"),
                model_quant=model_config.get("quant", ""),
                model_max_model_len=model_config.get("max_model_len", ""),
                model_gpu_memory_util=model_config.get("gpu_memory_util", ""),
                eval_max_tokens=config["evaluation"]["max_tokens"],
                eval_temperature=config["evaluation"]["temperature"],
                eval_language="", eval_question_id="", eval_run_number="",
                eval_image_ref="", eval_image_count="", eval_input_mode="",
            )

            if not vllm.start(model_config):
                print(f"❌ Failed to start vLLM for {model_config['name']}")
                gpu_monitor.set_context(phase="model_start_failed")
                vllm.stop()
                model_status["status"] = "startup_failed"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = "vLLM startup failed; see /tmp/vllm_<model>.log"
                model_statuses.append(model_status)
                continue

            evaluator: Optional[Evaluator] = None
            try:
                evaluator = Evaluator(vllm, config, items, image_root, gpu_monitor=gpu_monitor)
                results = evaluator.evaluate_model(model_name, model_config)
                all_results.append(results)
                model_status["status"] = "evaluated"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = (
                    f"successful_responses={results['successful_responses']}/"
                    f"{results['total_questions']} "
                    f"(skipped_missing_image={results['skipped_missing_image']})"
                )
                model_statuses.append(model_status)
            except Exception as e:
                model_status["status"] = "evaluation_error"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = f"{type(e).__name__}: {e}"
                model_statuses.append(model_status)
                raise
            finally:
                if evaluator:
                    evaluator.close()
                gpu_monitor.set_context(phase="stopping_model")
                vllm.stop()
                gpu_monitor.set_context(phase="idle")

        print("\n" + "="*60)
        print("VISION EVALUATION COMPLETE")
        print("="*60)
        print(f"\n📊 Summary:")
        for r in all_results:
            print(f"   {r['model_display_name']}: {r['successful_responses']}/{r['total_questions']}")

        failed = [m for m in model_statuses if m["status"] != "evaluated"]
        if failed:
            print("\n⚠️ Models not fully evaluated:")
            for m in failed:
                print(f"   - {m['model_name']}: {m['status']} ({m['details']})")

        db_path = Path(config["paths"]["results_dir"]) / "evaluation_results_euf_vision.db"
        print(f"\n💾 Results saved to: {db_path}")
        excel_path = export_results_to_excel(db_path)
        if excel_path:
            print(f"📄 Excel exported to: {excel_path}")
            print(f"📄 By-model Excel exported to: {db_path.with_name(f'{db_path.stem}_by_model.xlsx')}")

        status_path = run_paths["metadata_dir"] / "model_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(model_statuses, f, indent=2)
        print(f"📄 Model status: {status_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        vllm.stop()
        sys.exit(1)
    finally:
        gpu_monitor.stop()


if __name__ == "__main__":
    main()
