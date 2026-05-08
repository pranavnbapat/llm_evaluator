# gpu_runtime/evaluate_vision.py

#!/workspace/llm_evaluator/.venv/bin/python3
"""
Multimodal evaluation script for image and PDF tasks.

Forks evaluate_context.py: VLlmManager / GPUMonitor / DB scaffolding are kept,
but prompts are multimodal and the dataset is loaded from
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
from typing import Optional, Any
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
    pages_dir = run_dir / "media_pages"
    for p in [raw_dir, scores_dir, logs_dir, insights_dir, metadata_dir, pages_dir]:
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
        "pages_dir": pages_dir,
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


EU24_LANGUAGES = [
    "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR", "GA", "HR",
    "HU", "IT", "LT", "LV", "MT", "NL", "PL", "PT", "RO", "SK", "SL", "SV",
]


def _resolve_dataset_root(dataset_path: Path, raw_value: str, default_relative: str) -> Path:
    base = Path(raw_value or default_relative)
    if base.is_absolute():
        return base.resolve()
    candidate = (dataset_path.parent / base).resolve()
    if candidate.exists():
        return candidate
    return (REPO_ROOT / base).resolve()


def _normalize_languages(*language_sources) -> list[str]:
    normalized = []
    for source in language_sources:
        if not source:
            continue
        values = source if isinstance(source, list) else [source]
        for value in values:
            token = str(value).strip()
            if not token:
                continue
            up = token.upper()
            if up not in normalized:
                normalized.append(up)
    return normalized or ["EN"]


def _normalize_context(raw_context) -> list:
    if raw_context is None:
        return []
    if isinstance(raw_context, list):
        return raw_context
    return [raw_context]


def _localized_text(item: dict, base_key: str, language: str) -> str:
    direct_key = f"{base_key}_translations"
    translations = item.get(direct_key) or {}
    if isinstance(translations, dict):
        if language in translations:
            return str(translations[language]).strip()
        if language.upper() in translations:
            return str(translations[language.upper()]).strip()
        if "EN" in translations:
            return str(translations["EN"]).strip()
    value = item.get(base_key, "")
    return str(value).strip()


def _collect_media_refs(item: dict) -> list[str]:
    refs = []
    for key in ("image_url", "image_filename", "image_path"):
        value = item.get(key)
        if value:
            refs.append(str(value))
    for key in ("image_urls", "image_paths", "image_filenames"):
        values = item.get(key) or []
        if isinstance(values, list):
            refs.extend([str(v) for v in values if v])
    deduped = []
    for ref in refs:
        if ref not in deduped:
            deduped.append(ref)
    return deduped


def load_vision_dataset(dataset_path: Path) -> tuple[list[dict], dict]:
    """Load and normalize multimodal tasks from JSON."""
    with open(dataset_path) as f:
        data = json.load(f)

    dataset_languages = _normalize_languages(
        data.get("default_languages"),
        data.get("languages"),
    )
    image_root = _resolve_dataset_root(dataset_path, data.get("image_root", ""), "data/vision_images")
    pdf_root = _resolve_dataset_root(dataset_path, data.get("pdf_root", ""), "files")
    image_root.mkdir(parents=True, exist_ok=True)
    pdf_root.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for item in data.get("items", []):
        item_id = str(item.get("item_id") or item.get("document_id") or item.get("question_id") or f"item_{len(tasks)+1}")
        modality = str(item.get("modality") or ("pdf" if item.get("pdf_path") or item.get("pdf_filename") else "image")).strip().lower()
        task_type = str(item.get("task_type") or ("summary" if item.get("summary_prompt") else "qa")).strip().lower()
        item_context = _normalize_context(item.get("context"))
        item_expected_elements = item.get("expected_elements") or []
        item_reference_texts = item.get("reference_texts") or []
        item_reference_facts = item.get("reference_facts") or item.get("reference_answer") or []
        item_source_text = item.get("source_text", "") or ""
        item_max_sentences = int(item.get("max_sentences", 8 if task_type == "summary" else 6))
        item_pages_per_chunk = int(item.get("pages_per_chunk", item.get("page_batch_size", 3)))

        base_task = {
            "item_id": item_id,
            "modality": modality,
            "task_type": task_type,
            "context": item_context,
            "expected_elements": item_expected_elements,
            "reference_texts": item_reference_texts,
            "reference_facts": item_reference_facts,
            "source_text": item_source_text,
            "max_sentences": item_max_sentences,
            "pages_per_chunk": max(1, item_pages_per_chunk),
            "summary_prompt": item.get("summary_prompt") or item.get("prompt") or "",
            "media_refs": _collect_media_refs(item),
            "pdf_ref": item.get("pdf_path") or item.get("pdf_filename") or item.get("file_path") or "",
            "metadata": item.get("metadata") or {},
        }

        if item.get("questions"):
            for q in item["questions"]:
                question_id = str(q.get("question_id") or item.get("question_id") or f"{item_id}_qa")
                languages = _normalize_languages(
                    q.get("languages"),
                    q.get("language"),
                    item.get("languages"),
                    item.get("language"),
                    dataset_languages,
                )
                for language in languages:
                    question_text = (
                        _localized_text(q, "question", language)
                        or _localized_text(q, "prompt", language)
                    )
                    tasks.append({
                        **base_task,
                        "question_id": question_id,
                        "question_text": question_text,
                        "language": language,
                        "task_type": str(q.get("task_type") or task_type or "qa").lower(),
                        "expected_elements": q.get("expected_elements") or item_expected_elements,
                        "reference_texts": q.get("reference_texts") or item_reference_texts,
                        "reference_facts": q.get("reference_facts") or q.get("reference_answer") or item_reference_facts,
                        "source_text": q.get("source_text") or item_source_text,
                        "max_sentences": int(q.get("max_sentences", item_max_sentences)),
                    })
            continue

        languages = _normalize_languages(
            item.get("languages"),
            item.get("language"),
            dataset_languages,
        )
        question_id = str(item.get("question_id") or ("MM_SUMMARY" if task_type == "summary" else "MM_QA"))
        for language in languages:
            question_text = (
                _localized_text(item, "question", language)
                or _localized_text(item, "prompt", language)
                or _localized_text(item, "summary_prompt", language)
            )
            tasks.append({
                **base_task,
                "question_id": question_id,
                "question_text": question_text,
                "language": language,
            })

    return tasks, {
        "dataset": data,
        "dataset_path": dataset_path.resolve(),
        "image_root": image_root,
        "pdf_root": pdf_root,
    }


def resolve_image_refs(item: dict, image_root: Path) -> tuple[list[str], list[str]]:
    data_urls: list[str] = []
    display_refs: list[str] = []
    for ref in item.get("media_refs") or []:
        if ref.startswith(("http://", "https://", "data:")):
            data_urls.append(ref)
            display_refs.append(ref)
            continue
        p = Path(ref)
        if not p.is_absolute():
            p = (image_root / p).resolve()
        data_url = encode_image_as_data_url(p)
        if data_url:
            data_urls.append(data_url)
            display_refs.append(str(p))
    return data_urls, display_refs


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    existing = sorted(out_dir.glob("page-*.png"))
    if existing:
        return existing
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)], check=True)
    return sorted(out_dir.glob("page-*.png"))


def resolve_pdf_ref(item: dict, pdf_root: Path) -> Path:
    ref = str(item.get("pdf_ref") or "").strip()
    if not ref:
        raise FileNotFoundError("missing pdf_path/pdf_filename")
    p = Path(ref)
    if not p.is_absolute():
        p = (pdf_root / p).resolve()
    return p


def chunk_list(values: list, size: int) -> list[list]:
    return [values[i:i + size] for i in range(0, len(values), size)]


class Evaluator:
    """Multimodal evaluator for image QA/summarization and PDF QA/summarization."""

    def __init__(self, vllm: VLlmManager, config: dict, tasks: list[dict], asset_roots: dict,
                 pages_dir: Path, gpu_monitor: Optional[GPUMonitor] = None):
        self.vllm = vllm
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.tasks = tasks
        self.asset_roots = asset_roots
        self.pages_dir = pages_dir
        self.results_dir = Path(config["paths"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.results_dir / "evaluation_results_euf_vision.db"
        self._conn = self._open_db_connection()
        self._init_db()

        self.num_runs = int(config["evaluation"]["num_runs"])
        self.max_tokens = int(config["evaluation"]["max_tokens"])
        self.temperature = float(config["evaluation"]["temperature"])

    def _init_db(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                language TEXT,
                question_id TEXT,
                item_id TEXT,
                task_type TEXT,
                modality TEXT,
                run_number INTEGER,
                question_text TEXT,
                context TEXT,
                media_ref TEXT,
                media_count INTEGER,
                response TEXT,
                timestamp TEXT,
                latency_ms REAL,
                metadata_json TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                step_type TEXT,
                chunk_index INTEGER,
                page_start INTEGER,
                page_end INTEGER,
                prompt_text TEXT,
                response_text TEXT,
                latency_ms REAL,
                status TEXT,
                created_at TEXT
            )
        """)
        self._ensure_column("evaluations", "task_type", "TEXT")
        self._ensure_column("evaluations", "modality", "TEXT")
        self._ensure_column("evaluations", "media_ref", "TEXT")
        self._ensure_column("evaluations", "media_count", "INTEGER")
        self._ensure_column("evaluations", "metadata_json", "TEXT")
        self._conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, column_type: str) -> None:
        cursor = self._conn.cursor()
        columns = {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

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
            if isinstance(entry, str):
                parts.append(f"[{i}] {entry[:500]}")
                continue
            if not isinstance(entry, dict):
                continue
            title = entry.get("title", "")
            description = entry.get("description", "")
            subtitle = entry.get("subtitle", "")
            body = entry.get("text") or entry.get("content") or ""
            merged = ". ".join([p for p in [title, subtitle, description, body] if p]).strip()
            if merged:
                parts.append(f"[{i}] {merged[:700]}")
        return "\n\n".join(parts)

    def _set_monitor_context(self, *, phase: str, task: dict, run_number: int, media_ref: str, media_count: int, input_mode: str) -> None:
        if not self.gpu_monitor:
            return
        self.gpu_monitor.set_context(
            phase=phase,
            eval_language=task["language"],
            eval_question_id=task["question_id"],
            eval_run_number=run_number,
            eval_image_ref=media_ref,
            eval_image_count=media_count,
            eval_input_mode=input_mode,
        )

    def _call_multimodal(self, content_parts: list, *, max_tokens: Optional[int] = None) -> tuple[Optional[str], float]:
        started = time.time()
        response = self.vllm.chat_completion_multimodal(
            content_parts=content_parts,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        latency_ms = (time.time() - started) * 1000
        return response, latency_ms

    def _build_media_question_prompt(self, task: dict) -> str:
        context_block = self._format_context(task.get("context") or [])
        context_text = f"OPTIONAL SUPPORTING CONTEXT:\n{context_block}\n\n" if context_block else ""
        modality_name = "document pages" if task["modality"] == "pdf" else "image"
        if task["task_type"] == "summary":
            custom = task.get("question_text") or task.get("summary_prompt") or "Summarize the attached content."
            return (
                f"You are reviewing attached {modality_name}. {context_text}"
                f"TASK ({task['language']}):\n{custom}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Answer in {task['language']}.\n"
                f"2. Summarize only what is supported by the visible content.\n"
                f"3. Do not invent details not visible in the media.\n"
                f"4. Keep the answer to {task.get('max_sentences', 8)} sentences or fewer.\n\n"
                f"Your summary:"
            )
        return (
            f"You are an expert agriculture advisor reviewing attached {modality_name}. {context_text}"
            f"QUESTION ({task['language']}):\n{task['question_text']}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer in {task['language']}.\n"
            f"2. Use only what is visible in the media plus the optional supporting context.\n"
            f"3. If the media does not support a claim, say so plainly instead of guessing.\n"
            f"4. Be practical and concise.\n\n"
            f"Your answer:"
        )

    def _build_pdf_map_prompt(self, task: dict, page_start: int, page_end: int, total_pages: int) -> str:
        if task["task_type"] == "summary":
            base = task.get("question_text") or task.get("summary_prompt") or "Summarize the attached document pages."
            return (
                f"You are reading pages {page_start}-{page_end} of {total_pages} of a PDF document.\n"
                f"TASK ({task['language']}): {base}\n\n"
                f"Write a faithful partial summary of only these pages in {task['language']}. "
                f"Capture headings, figures, facts, recommendations, and any structured sections. "
                f"Do not speculate about pages you have not seen."
            )
        return (
            f"You are reading pages {page_start}-{page_end} of {total_pages} of a PDF document.\n"
            f"QUESTION ({task['language']}): {task['question_text']}\n\n"
            f"Extract only the evidence from these pages that helps answer the question. "
            f"If the pages do not contain useful evidence, reply exactly NO_EVIDENCE. "
            f"Keep the result concise and factual in {task['language']}."
        )

    def _build_pdf_reduce_prompt(self, task: dict, snippets: list[str], total_pages: int) -> str:
        joined = "\n\n".join(f"[chunk {i+1}] {text}" for i, text in enumerate(snippets))
        context_block = self._format_context(task.get("context") or [])
        optional_context = f"\n\nOPTIONAL SUPPORTING CONTEXT:\n{context_block}" if context_block else ""
        if task["task_type"] == "summary":
            return (
                f"You have partial summaries for a {total_pages}-page PDF document.{optional_context}\n\n"
                f"PARTIAL SUMMARIES:\n{joined}\n\n"
                f"Synthesize them into one coherent summary in {task['language']}. "
                f"Keep the output within {task.get('max_sentences', 8)} sentences. "
                f"Do not invent any content that is not supported by the partial summaries."
            )
        return (
            f"You have extracted evidence snippets from a {total_pages}-page PDF document.{optional_context}\n\n"
            f"QUESTION ({task['language']}): {task['question_text']}\n\n"
            f"EVIDENCE SNIPPETS:\n{joined}\n\n"
            f"Write the final answer in {task['language']}. "
            f"Use only the evidence above plus the optional supporting context. "
            f"If the evidence is insufficient, say that clearly."
        )

    def _task_run(self, task: dict) -> tuple[Optional[str], float, str, int, dict]:
        try:
            if task["modality"] == "pdf":
                return self._run_pdf_task(task)
            return self._run_image_task(task)
        except FileNotFoundError as e:
            return None, 0.0, str(e), 0, {"status": "missing_media"}
        except subprocess.CalledProcessError as e:
            return None, 0.0, str(e), 0, {"status": "media_processing_failed"}

    def _run_image_task(self, task: dict) -> tuple[Optional[str], float, str, int, dict]:
        image_urls, display_refs = resolve_image_refs(task, self.asset_roots["image_root"])
        if not image_urls:
            return None, 0.0, "", 0, {"status": "missing_media"}
        content_parts = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
        content_parts.append({"type": "text", "text": self._build_media_question_prompt(task)})
        response, latency_ms = self._call_multimodal(content_parts)
        return response, latency_ms, "\n".join(display_refs), len(image_urls), {"status": "ok", "chunks": 1}

    def _run_pdf_task(self, task: dict) -> tuple[Optional[str], float, str, int, dict]:
        pdf_path = resolve_pdf_ref(task, self.asset_roots["pdf_root"])
        if not pdf_path.exists():
            return None, 0.0, str(pdf_path), 0, {"status": "missing_media"}

        pages_cache = self.pages_dir / task["item_id"] / task["language"]
        page_paths = render_pdf_to_pngs(pdf_path, pages_cache)
        if not page_paths:
            return None, 0.0, str(pdf_path), 0, {"status": "no_pages"}

        chunks = chunk_list(page_paths, task.get("pages_per_chunk", 3))
        snippets: list[str] = []
        steps: list[dict] = []
        total_latency = 0.0

        for chunk_index, chunk_paths in enumerate(chunks, start=1):
            page_start = ((chunk_index - 1) * task.get("pages_per_chunk", 3)) + 1
            page_end = min(page_start + len(chunk_paths) - 1, len(page_paths))
            content_parts = []
            for p in chunk_paths:
                data_url = encode_image_as_data_url(p)
                if data_url:
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            if not content_parts:
                continue
            content_parts.append({
                "type": "text",
                "text": self._build_pdf_map_prompt(task, page_start, page_end, len(page_paths)),
            })
            response, latency_ms = self._call_multimodal(content_parts)
            total_latency += latency_ms
            clean_response = (response or "").strip()
            status = "ok" if clean_response else "empty"
            steps.append({
                "step_type": "map",
                "chunk_index": chunk_index,
                "page_start": page_start,
                "page_end": page_end,
                "prompt_text": content_parts[-1]["text"],
                "response_text": clean_response,
                "latency_ms": latency_ms,
                "status": status,
            })
            if clean_response and clean_response != "NO_EVIDENCE":
                snippets.append(clean_response)

        if not snippets:
            return None, total_latency, str(pdf_path), len(page_paths), {"status": "no_evidence", "steps": steps}

        if len(snippets) == 1:
            final_response = snippets[0]
        else:
            reduce_prompt = self._build_pdf_reduce_prompt(task, snippets, len(page_paths))
            reduce_response, reduce_latency_ms = self._call_multimodal(
                [{"type": "text", "text": reduce_prompt}],
                max_tokens=self.max_tokens,
            )
            total_latency += reduce_latency_ms
            final_response = (reduce_response or "").strip()
            steps.append({
                "step_type": "reduce",
                "chunk_index": len(chunks) + 1,
                "page_start": 1,
                "page_end": len(page_paths),
                "prompt_text": reduce_prompt,
                "response_text": final_response,
                "latency_ms": reduce_latency_ms,
                "status": "ok" if final_response else "empty",
            })

        return final_response or None, total_latency, str(pdf_path), len(page_paths), {
            "status": "ok" if final_response else "empty",
            "steps": steps,
            "page_count": len(page_paths),
            "chunk_count": len(chunks),
        }

    def evaluate_model(self, model_name: str, model_config: dict) -> dict:
        print(f"\n{'='*60}")
        print(f"  Evaluating (multimodal): {model_config['name']}")
        print(f"  Database: evaluation_results_euf_vision.db")
        print(f"{'='*60}")

        results = {
            "model_name": model_name,
            "model_display_name": model_config["name"],
            "timestamp": datetime.now().isoformat(),
            "total_tasks": 0,
            "successful_responses": 0,
            "skipped_missing_media": 0,
        }

        for task in tqdm(self.tasks, desc="Tasks"):
            for run in range(1, self.num_runs + 1):
                input_mode = f"{task['modality']}+text"
                media_hint = task.get("pdf_ref") or "\n".join(task.get("media_refs") or [])
                self._set_monitor_context(
                    phase="evaluating",
                    task=task,
                    run_number=run,
                    media_ref=media_hint,
                    media_count=0,
                    input_mode=input_mode,
                )
                response, latency_ms, media_ref, media_count, metadata = self._task_run(task)
                if metadata.get("status") in {"missing_media", "media_processing_failed"}:
                    print(f"   ⚠️ Skipping {task['item_id']}:{task['question_id']}:{task['language']}: {metadata['status']}")
                    results["skipped_missing_media"] += 1
                    results["total_tasks"] += 1
                    continue
                if response:
                    evaluation_id = self._save_result(
                        model_name=model_name,
                        language=task["language"],
                        question_id=task["question_id"],
                        item_id=task["item_id"],
                        task_type=task["task_type"],
                        modality=task["modality"],
                        run_number=run,
                        question_text=task["question_text"],
                        context=task.get("context") or [],
                        media_ref=media_ref,
                        media_count=media_count,
                        response=response,
                        latency_ms=latency_ms,
                        metadata_json=metadata,
                    )
                    self._save_steps(evaluation_id, metadata.get("steps") or [])
                    results["successful_responses"] += 1
                results["total_tasks"] += 1

        if self.gpu_monitor:
            self.gpu_monitor.set_context(
                eval_language="", eval_question_id="", eval_run_number="",
                eval_image_ref="", eval_image_count="", eval_input_mode="",
            )

        json_path = self.results_dir / f"{model_name}_vision_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n📊 Results: {results['successful_responses']}/{results['total_tasks']} successful")
        if results["skipped_missing_media"]:
            print(f"🖼️  Skipped (missing media): {results['skipped_missing_media']}")
        print(f"💾 Saved to: {self.db_path}")
        print(f"📄 JSON: {json_path}")
        return results

    def _save_result(self, *, model_name, language, question_id, item_id, task_type, modality, run_number,
                     question_text, context, media_ref, media_count, response, latency_ms, metadata_json):
        context_json = json.dumps(context) if context else ""
        metadata_text = json.dumps(metadata_json) if metadata_json else ""
        for attempt in range(3):
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                cursor = self._conn.cursor()
                cursor.execute("""
                    INSERT INTO evaluations
                    (model_name, language, question_id, item_id, task_type, modality, run_number, question_text, context,
                     media_ref, media_count, response, timestamp, latency_ms, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, language, question_id, item_id, task_type, modality, run_number,
                    question_text, context_json, media_ref, media_count, response,
                    datetime.now().isoformat(), latency_ms, metadata_text,
                ))
                self._conn.commit()
                return int(cursor.lastrowid)
            except sqlite3.OperationalError as e:
                if "unable to open database file" in str(e).lower() and attempt < 2:
                    print(f"⚠️ SQLite open failure ({self.db_path}), retrying {attempt + 1}/2...")
                    time.sleep(1.0)
                    self._reconnect_db()
                    continue
                raise

    def _save_steps(self, evaluation_id: int, steps: list[dict]) -> None:
        if not steps:
            return
        cursor = self._conn.cursor()
        for step in steps:
            cursor.execute("""
                INSERT INTO evaluation_steps
                (evaluation_id, step_type, chunk_index, page_start, page_end, prompt_text, response_text,
                 latency_ms, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                step.get("step_type", ""),
                step.get("chunk_index"),
                step.get("page_start"),
                step.get("page_end"),
                step.get("prompt_text", ""),
                step.get("response_text", ""),
                step.get("latency_ms"),
                step.get("status", ""),
                datetime.now().isoformat(),
            ))
        self._conn.commit()


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
    print("EU-FarmBook Multimodal Evaluation")
    print("VLM Benchmarking with Image and PDF Tasks")
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

    dataset_env = os.getenv("EVAL_VISION_DATASET", "").strip()
    dataset_path = Path(dataset_env).expanduser() if dataset_env else (REPO_ROOT / "data" / "evaluation_vision_questions.json")
    if not dataset_path.exists():
        print(f"❌ Vision dataset not found: {dataset_path}")
        sys.exit(2)
    tasks, asset_roots = load_vision_dataset(dataset_path)
    print(
        f"📚 Dataset: {dataset_path}  ({len(tasks)} tasks, "
        f"image_root={asset_roots['image_root']}, pdf_root={asset_roots['pdf_root']})"
    )

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
        "image_root": str(asset_roots["image_root"]),
        "pdf_root": str(asset_roots["pdf_root"]),
        "evaluation_mode": "multimodal",
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
    print(f"\n🧪 Tasks: {len(tasks)}  ×  runs: {config['evaluation']['num_runs']}")
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
                evaluator = Evaluator(
                    vllm,
                    config,
                    tasks,
                    asset_roots,
                    run_paths["pages_dir"],
                    gpu_monitor=gpu_monitor,
                )
                results = evaluator.evaluate_model(model_name, model_config)
                all_results.append(results)
                model_status["status"] = "evaluated"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = (
                    f"successful_responses={results['successful_responses']}/"
                    f"{results['total_tasks']} "
                    f"(skipped_missing_media={results['skipped_missing_media']})"
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
            print(f"   {r['model_display_name']}: {r['successful_responses']}/{r['total_tasks']}")

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
