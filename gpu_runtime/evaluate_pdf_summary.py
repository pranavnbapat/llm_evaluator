# gpu_runtime/evaluate_pdf_summary.py

#!/workspace/llm_evaluator/.venv/bin/python3
"""
PDF map-reduce summarisation for VLMs.

For each PDF in `files/`:
  1. Render pages to PNG (pdftoppm, 150 dpi). Cached per run.
  2. MAP   : split pages into batches of N (default 3), send each batch to the
            VLM as a multimodal message, capture batch summary + time.
  3. REDUCE: feed all batch summaries (text only) back to the VLM, capture an
            overall summary + time. If a PDF has only one batch, the batch
            summary is reused as the overall summary.

Iterates the model loop the same way as evaluate_context.py / evaluate_vision.py:
warm vLLM with one model, process every PDF, stop, advance.

Database: evaluation_pdf_summaries.db
  - pdf_runs    : one row per (model, file)
  - pdf_batches : one row per (run, batch)
"""
import os
import re
import sys
import json
import time
import base64
import sqlite3
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import yaml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_vision import VLlmManager, GPUMonitor  # noqa: E402


def detect_gpu_bucket() -> tuple[str, str]:
    override = os.getenv("EVAL_RUN_GPU", "").strip().lower()
    if override:
        safe = re.sub(r"[^a-z0-9_\\-]+", "_", override).strip("_")
        return safe or "unknown_gpu", "env:EVAL_RUN_GPU"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        names = [ln.strip().lower() for ln in result.stdout.splitlines() if ln.strip()]
        name = names[0] if names else ""
        if "b200" in name or "gb200" in name: return "b200", f"nvidia-smi:{name}"
        if "h200" in name: return "h200_sxm", f"nvidia-smi:{name}"
        if "h100" in name: return "h100_sxm", f"nvidia-smi:{name}"
        if "l40s" in name or "l40" in name: return "l40s", f"nvidia-smi:{name}"
        if "3090" in name: return "3090", f"nvidia-smi:{name}"
        if "a100" in name: return "a100", f"nvidia-smi:{name}"
        if "a40" in name: return "a40", f"nvidia-smi:{name}"
        safe = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        return (safe or "unknown_gpu"), f"nvidia-smi:{name or 'unknown'}"
    except Exception:
        return "unknown_gpu", "fallback:unknown"


def resolve_run_paths(base_results_dir: Path) -> dict:
    run_dir_override = os.getenv("EVAL_RUN_DIR", "").strip()
    gpu_bucket, gpu_source = detect_gpu_bucket()
    run_id = os.getenv("EVAL_RUN_ID", "").strip()
    if not run_id:
        run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_pdf_eval"
    if run_dir_override:
        run_dir = Path(run_dir_override).expanduser().resolve()
        run_id = run_dir.name
        gpu_bucket = run_dir.parent.name if run_dir.parent.name else gpu_bucket
        run_source = "env:EVAL_RUN_DIR"
    else:
        run_dir = (base_results_dir / "runs" / gpu_bucket / run_id).resolve()
        run_source = "auto"

    raw_dir = run_dir / "raw"
    logs_dir = run_dir / "logs"
    pages_dir = run_dir / "pdf_pages"
    metadata_dir = run_dir / "metadata"
    for p in [raw_dir, logs_dir, pages_dir, metadata_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "base_results_dir": base_results_dir.resolve(),
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "logs_dir": logs_dir,
        "pages_dir": pages_dir,
        "metadata_dir": metadata_dir,
        "run_id": run_id,
        "gpu_bucket": gpu_bucket,
        "gpu_source": gpu_source,
        "run_source": run_source,
    }


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            if key not in os.environ:
                os.environ[key] = value


def substitute_env_vars(value):
    if isinstance(value, str):
        def repl(m):
            expr = m.group(1)
            if ":-" in expr:
                name, default = expr.split(":-", 1)
                return os.environ.get(name, default)
            return os.environ.get(expr, "")
        return re.sub(r"\$\{([^}]+)\}", repl, value)
    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_env_vars(v) for v in value]
    return value


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    env_path = Path(__file__).parent / ".env"
    load_env_file(env_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return substitute_env_vars(cfg)


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Path]:
    """Render each page of pdf_path to a PNG using pdftoppm. Returns sorted list of PNG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    existing = sorted(out_dir.glob("page-*.png"))
    if existing:
        return existing
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("page-*.png"))


def encode_png_as_data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def chunk(seq: List, size: int) -> List[List]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def build_batch_prompt(file_name: str, page_start: int, page_end: int, total_pages: int) -> str:
    return (
        f"You are reading pages {page_start}-{page_end} of {total_pages} of the document '{file_name}'. "
        f"Read the {page_end - page_start + 1} attached page image(s) and write a self-contained summary "
        f"of what these pages cover in 4-6 sentences. Capture key facts, headings, figures, and any "
        f"actionable items. Do NOT speculate about pages you have not seen."
    )


def build_reduce_prompt(file_name: str, total_pages: int, batch_summaries: List[str]) -> str:
    joined = "\n\n".join(
        f"[batch {i+1}] {s}" for i, s in enumerate(batch_summaries)
    )
    return (
        f"You have been given partial summaries for the document '{file_name}' "
        f"({total_pages} pages total), produced by reading pages in batches.\n\n"
        f"PARTIAL SUMMARIES:\n{joined}\n\n"
        f"Synthesise these into ONE coherent overall summary of the document in 8-12 sentences. "
        f"Cover the document's purpose, structure, and main points. Do not invent content that is "
        f"not supported by the partial summaries above."
    )


class PdfEvaluator:
    def __init__(self, vllm: VLlmManager, config: dict, raw_dir: Path,
                 batch_size: int, gpu_monitor: Optional[GPUMonitor] = None):
        self.vllm = vllm
        self.config = config
        self.raw_dir = raw_dir
        self.batch_size = batch_size
        self.gpu_monitor = gpu_monitor
        self.db_path = raw_dir / "evaluation_pdf_summaries.db"
        self._conn = self._open_db()
        self._init_db()
        self.max_tokens = int(config["evaluation"]["max_tokens"])
        self.temperature = float(config["evaluation"]["temperature"])

    def _open_db(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        c = self._conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS pdf_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                file_name TEXT,
                total_pages INTEGER,
                batch_size INTEGER,
                num_batches INTEGER,
                overall_summary TEXT,
                overall_summary_time_ms REAL,
                map_total_time_ms REAL,
                total_time_ms REAL,
                started_at TEXT,
                finished_at TEXT,
                status TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS pdf_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES pdf_runs(id),
                batch_index INTEGER,
                page_start INTEGER,
                page_end INTEGER,
                num_pages INTEGER,
                summary TEXT,
                batch_time_ms REAL,
                status TEXT
            )
        """)
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def _insert_run_stub(self, model_name: str, file_name: str, total_pages: int,
                         num_batches: int, started_at: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO pdf_runs(model_name, file_name, total_pages, batch_size, "
            "num_batches, started_at, status) VALUES (?, ?, ?, ?, ?, ?, 'running')",
            (model_name, file_name, total_pages, self.batch_size, num_batches, started_at),
        )
        self._conn.commit()
        return cur.lastrowid

    def _insert_batch(self, run_id: int, batch_index: int, page_start: int,
                      page_end: int, summary: Optional[str], batch_time_ms: float, status: str):
        self._conn.execute(
            "INSERT INTO pdf_batches(run_id, batch_index, page_start, page_end, "
            "num_pages, summary, batch_time_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, batch_index, page_start, page_end, page_end - page_start + 1,
             summary, batch_time_ms, status),
        )
        self._conn.commit()

    def _finalise_run(self, run_id: int, overall_summary: Optional[str],
                      overall_time_ms: float, map_total_ms: float, total_ms: float, status: str):
        self._conn.execute(
            "UPDATE pdf_runs SET overall_summary=?, overall_summary_time_ms=?, "
            "map_total_time_ms=?, total_time_ms=?, finished_at=?, status=? WHERE id=?",
            (overall_summary, overall_time_ms, map_total_ms, total_ms,
             datetime.now().isoformat(), status, run_id),
        )
        self._conn.commit()

    def evaluate_pdf(self, model_name: str, model_display_name: str, pdf_path: Path,
                     page_pngs: List[Path]) -> dict:
        file_name = pdf_path.name
        total_pages = len(page_pngs)
        batches = chunk(list(range(1, total_pages + 1)), self.batch_size)
        num_batches = len(batches)
        started_at = datetime.now().isoformat()
        run_id = self._insert_run_stub(model_name, file_name, total_pages, num_batches, started_at)

        print(f"\n  📄 {file_name}  | pages={total_pages}  | batches={num_batches} (size={self.batch_size})")

        wall_start = time.time()
        batch_summaries: List[str] = []
        map_total_ms = 0.0
        any_batch_failed = False

        for bidx, page_nums in enumerate(batches, start=1):
            page_start, page_end = page_nums[0], page_nums[-1]
            page_paths = [page_pngs[p - 1] for p in page_nums]

            content_parts = [{"type": "text", "text": build_batch_prompt(
                file_name, page_start, page_end, total_pages)}]
            for pp in page_paths:
                content_parts.append({"type": "image_url",
                                      "image_url": {"url": encode_png_as_data_url(pp)}})

            if self.gpu_monitor:
                self.gpu_monitor.set_context(
                    phase="map_batch",
                    eval_question_id=file_name,
                    eval_run_number=bidx,
                    eval_image_count=len(page_paths),
                    eval_input_mode="pdf_pages",
                )

            t0 = time.time()
            summary = self.vllm.chat_completion_multimodal(
                content_parts=content_parts,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            batch_ms = (time.time() - t0) * 1000.0
            map_total_ms += batch_ms

            if summary:
                batch_summaries.append(summary)
                self._insert_batch(run_id, bidx, page_start, page_end, summary, batch_ms, "ok")
                print(f"    ✅ batch {bidx}/{num_batches} (pages {page_start}-{page_end}) in {batch_ms:.0f} ms")
            else:
                any_batch_failed = True
                self._insert_batch(run_id, bidx, page_start, page_end, None, batch_ms, "failed")
                print(f"    ❌ batch {bidx}/{num_batches} failed in {batch_ms:.0f} ms")

        # Reduce
        overall_summary: Optional[str] = None
        overall_ms = 0.0
        if batch_summaries:
            if num_batches == 1:
                overall_summary = batch_summaries[0]
                overall_ms = 0.0
                print(f"    ↪️  single batch — using batch summary as overall")
            else:
                if self.gpu_monitor:
                    self.gpu_monitor.set_context(
                        phase="reduce",
                        eval_question_id=file_name,
                        eval_run_number=0,
                        eval_image_count=0,
                        eval_input_mode="text_reduce",
                    )
                reduce_prompt = build_reduce_prompt(file_name, total_pages, batch_summaries)
                t0 = time.time()
                overall_summary = self.vllm.chat_completion_multimodal(
                    content_parts=[{"type": "text", "text": reduce_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                overall_ms = (time.time() - t0) * 1000.0
                if overall_summary:
                    print(f"    ✅ overall summary in {overall_ms:.0f} ms")
                else:
                    print(f"    ❌ overall summary failed after {overall_ms:.0f} ms")

        total_ms = (time.time() - wall_start) * 1000.0
        if not batch_summaries:
            status = "all_batches_failed"
        elif any_batch_failed or (num_batches > 1 and not overall_summary):
            status = "partial"
        else:
            status = "ok"
        self._finalise_run(run_id, overall_summary, overall_ms, map_total_ms, total_ms, status)

        return {
            "run_id": run_id,
            "model_name": model_name,
            "model_display_name": model_display_name,
            "file_name": file_name,
            "total_pages": total_pages,
            "num_batches": num_batches,
            "batch_size": self.batch_size,
            "map_total_time_ms": map_total_ms,
            "overall_summary_time_ms": overall_ms,
            "total_time_ms": total_ms,
            "status": status,
            "batch_summaries": batch_summaries,
            "overall_summary": overall_summary,
        }


def export_to_excel(db_path: Path) -> Optional[Path]:
    if not db_path.exists():
        return None
    excel_path = db_path.with_suffix(".xlsx")
    conn = sqlite3.connect(db_path)
    try:
        runs = pd.read_sql_query("SELECT * FROM pdf_runs ORDER BY id", conn)
        batches = pd.read_sql_query("SELECT * FROM pdf_batches ORDER BY id", conn)
        if runs.empty and batches.empty:
            return None
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as w:
            runs.to_excel(w, sheet_name="pdf_runs", index=False)
            batches.to_excel(w, sheet_name="pdf_batches", index=False)
        return excel_path
    finally:
        conn.close()


def main():
    print("=" * 60)
    print("EU-FarmBook PDF Map-Reduce Summarisation Evaluation")
    print("=" * 60)

    config = load_config()

    files_dir = Path(os.getenv("EVAL_PDF_DIR", "")).expanduser()
    if not str(files_dir):
        files_dir = REPO_ROOT / "files"
    if not files_dir.exists():
        print(f"❌ Files directory not found: {files_dir}")
        sys.exit(2)
    pdfs = sorted(files_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in: {files_dir}")
        sys.exit(2)

    batch_size = int(os.getenv(
        "EVAL_PDF_BATCH_SIZE",
        config.get("evaluation", {}).get("pdf_batch_size", 3),
    ))
    if batch_size < 1:
        print("❌ batch_size must be >= 1")
        sys.exit(2)
    dpi = int(os.getenv("EVAL_PDF_DPI", "150"))

    base_results_dir = Path(config["paths"]["results_dir"]).resolve()
    try:
        run_paths = resolve_run_paths(base_results_dir)
    except Exception as e:
        fallback = (REPO_ROOT / "results").resolve()
        print(f"⚠️ Could not create run path under '{base_results_dir}': {e}")
        print(f"   Falling back to: {fallback}")
        run_paths = resolve_run_paths(fallback)
    config["paths"]["results_dir"] = str(run_paths["raw_dir"])

    print(f"\n📁 PDFs from: {files_dir}  ({len(pdfs)} file(s))")
    print(f"📦 Batch size: {batch_size}   |   DPI: {dpi}")
    print(f"🧭 Run dir:   {run_paths['run_dir']}")
    print(f"💾 DB:        {run_paths['raw_dir'] / 'evaluation_pdf_summaries.db'}")

    print(f"\n🖨️  Rendering PDFs to PNGs (cached per file)...")
    pdf_pages: dict = {}
    for pdf in pdfs:
        page_dir = run_paths["pages_dir"] / pdf.stem
        try:
            pages = render_pdf_to_pngs(pdf, page_dir, dpi=dpi)
        except subprocess.CalledProcessError as e:
            print(f"   ❌ pdftoppm failed for {pdf.name}: {e}")
            continue
        if not pages:
            print(f"   ⚠️ No pages rendered for {pdf.name}")
            continue
        pdf_pages[pdf] = pages
        print(f"   ✅ {pdf.name}: {len(pages)} page(s) → {page_dir}")
    if not pdf_pages:
        print("❌ No usable PDFs after rendering. Aborting.")
        sys.exit(2)

    run_meta = {
        "created_at": datetime.now().isoformat(),
        "run_id": run_paths["run_id"],
        "run_dir": str(run_paths["run_dir"]),
        "raw_dir": str(run_paths["raw_dir"]),
        "logs_dir": str(run_paths["logs_dir"]),
        "pages_dir": str(run_paths["pages_dir"]),
        "gpu_bucket": run_paths["gpu_bucket"],
        "gpu_detected_from": run_paths["gpu_source"],
        "run_path_source": run_paths["run_source"],
        "evaluation_mode": "pdf_summary",
        "files_dir": str(files_dir),
        "batch_size": batch_size,
        "dpi": dpi,
        "files": [{"name": p.name, "pages": len(pdf_pages[p])} for p in pdf_pages],
    }
    with open(run_paths["metadata_dir"] / "run_info.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    vllm = VLlmManager(config)
    gpu_monitor = GPUMonitor(run_paths["logs_dir"] / "gpu_metrics.csv")

    models = list(config["models"].values())
    print(f"\n📋 Models to evaluate: {len(models)}")
    for m in models:
        suffix = "  (+extra)" if m.get("vllm_extra_args") else ""
        print(f"   - {m['name']}{suffix}")

    all_results: List[dict] = []
    model_statuses: List[dict] = []

    try:
        gpu_monitor.start()
        for model_config in models:
            model_name = model_config["name"].replace(" ", "_").lower()
            display = model_config.get("name", model_name)
            model_status = {
                "model_name": display,
                "repo": model_config.get("repo", ""),
                "started_at": datetime.now().isoformat(),
                "finished_at": None,
                "status": "pending",
                "details": "",
                "per_file_total_ms": {},
                "model_grand_total_ms": 0.0,
            }
            gpu_monitor.set_context(
                phase="loading_model",
                model_name=display,
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
                print(f"❌ Failed to start vLLM for {display}")
                gpu_monitor.set_context(phase="model_start_failed")
                vllm.stop()
                model_status["status"] = "startup_failed"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = "vLLM startup failed; see /tmp/vllm_<model>.log"
                model_statuses.append(model_status)
                continue

            evaluator: Optional[PdfEvaluator] = None
            try:
                evaluator = PdfEvaluator(
                    vllm=vllm, config=config, raw_dir=run_paths["raw_dir"],
                    batch_size=batch_size, gpu_monitor=gpu_monitor,
                )
                model_grand_total_ms = 0.0
                model_results = []
                for pdf_path, page_pngs in pdf_pages.items():
                    res = evaluator.evaluate_pdf(model_name, display, pdf_path, page_pngs)
                    model_results.append(res)
                    model_grand_total_ms += res["total_time_ms"]
                    model_status["per_file_total_ms"][pdf_path.name] = res["total_time_ms"]
                model_status["model_grand_total_ms"] = model_grand_total_ms

                ok = sum(1 for r in model_results if r["status"] == "ok")
                model_status["status"] = "evaluated" if ok == len(model_results) else "partial"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = (
                    f"files_ok={ok}/{len(model_results)} | "
                    f"grand_total={model_grand_total_ms:.0f} ms"
                )

                json_path = run_paths["raw_dir"] / f"{model_name}_pdf_{datetime.now():%Y%m%d_%H%M%S}.json"
                with open(json_path, "w") as f:
                    json.dump({
                        "model_name": model_name,
                        "model_display_name": display,
                        "model_grand_total_ms": model_grand_total_ms,
                        "files": model_results,
                    }, f, indent=2)
                print(f"\n  📄 JSON written: {json_path}")
                all_results.extend(model_results)

            except Exception as e:
                model_status["status"] = "evaluation_error"
                model_status["finished_at"] = datetime.now().isoformat()
                model_status["details"] = f"{type(e).__name__}: {e}"
                model_statuses.append(model_status)
                raise
            else:
                model_statuses.append(model_status)
            finally:
                if evaluator:
                    evaluator.close()
                gpu_monitor.set_context(phase="stopping_model")
                vllm.stop()
                gpu_monitor.set_context(phase="idle")

        # Wrap up
        print("\n" + "=" * 60)
        print("PDF SUMMARY EVALUATION COMPLETE")
        print("=" * 60)
        for m in model_statuses:
            print(f"  {m['model_name']}: {m['status']} | {m['details']}")

        db_path = Path(config["paths"]["results_dir"]) / "evaluation_pdf_summaries.db"
        excel = export_to_excel(db_path)
        if excel:
            print(f"\n📄 Excel: {excel}")
        status_path = run_paths["metadata_dir"] / "model_status.json"
        with open(status_path, "w") as f:
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
