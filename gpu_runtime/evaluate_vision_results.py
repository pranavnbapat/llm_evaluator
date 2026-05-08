#!/usr/bin/env python3
"""
Evaluate multimodal image/PDF responses from SQLite database using scientific metrics.

Usage:
    python gpu_runtime/evaluate_vision_results.py
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics.scientific_metrics import ResponseEvaluator
from evaluate_vision import load_vision_dataset


def detect_gpu_bucket() -> tuple[str, str]:
    override = os.getenv("EVAL_RUN_GPU", "").strip().lower()
    if override:
        safe = re.sub(r"[^a-z0-9_\-]+", "_", override).strip("_")
        return safe or "unknown_gpu", "env:EVAL_RUN_GPU"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        name = ""
        for ln in result.stdout.splitlines():
            if ln.strip():
                name = ln.strip().lower()
                break
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
        return safe or "unknown_gpu", f"nvidia-smi:{name or 'unknown'}"
    except Exception:
        return "unknown_gpu", "fallback:unknown"


def resolve_scoring_paths(base_results_dir: Path) -> dict:
    run_dir_env = os.getenv("EVAL_RUN_DIR", "").strip()
    run_id_env = os.getenv("EVAL_RUN_ID", "").strip()
    gpu_bucket, gpu_source = detect_gpu_bucket()

    run_dir = None
    source = "legacy"
    if run_dir_env:
        run_dir = Path(run_dir_env).expanduser().resolve()
        source = "env:EVAL_RUN_DIR"
    elif run_id_env:
        run_dir = (base_results_dir / "runs" / gpu_bucket / run_id_env).resolve()
        source = "env:EVAL_RUN_ID"
    else:
        latest_link = base_results_dir / "latest" / gpu_bucket
        if latest_link.exists() or latest_link.is_symlink():
            try:
                run_dir = latest_link.resolve()
                source = f"latest:{gpu_bucket}"
            except Exception:
                run_dir = None

    if run_dir:
        raw_dir = run_dir / "raw"
        scores_dir = run_dir / "scores"
        metadata_dir = run_dir / "metadata"
        raw_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        db_path = raw_dir / "evaluation_results_euf_vision.db"
        scores_db_path = scores_dir / "evaluation_scores_euf_vision.db"
        scores_xlsx_path = scores_dir / "evaluation_scores_euf_vision.xlsx"
    else:
        db_path = base_results_dir / "evaluation_results_euf_vision.db"
        scores_db_path = base_results_dir / "evaluation_scores_euf_vision.db"
        scores_xlsx_path = base_results_dir / "evaluation_scores_euf_vision.xlsx"
        raw_dir = base_results_dir
        scores_dir = base_results_dir
        metadata_dir = base_results_dir

    return {
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "scores_dir": scores_dir,
        "metadata_dir": metadata_dir,
        "db_path": db_path,
        "scores_db_path": scores_db_path,
        "scores_xlsx_path": scores_xlsx_path,
        "gpu_bucket": gpu_bucket,
        "gpu_source": gpu_source,
        "source": source,
    }


def export_scores_to_excel(scores_db_path: Path, out_xlsx_path: Path) -> bool:
    try:
        import pandas as pd
    except ImportError:
        print("\n⚠️ Skipping Excel export: pandas/openpyxl not installed.")
        return False

    conn = sqlite3.connect(scores_db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM scores", conn)
    finally:
        conn.close()

    with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="scores", index=False)
    return True


def _read_run_dataset_path(run_dir: Path) -> Path | None:
    info_path = run_dir / "metadata" / "run_info.json"
    if not info_path.exists():
        return None
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    dataset_path = payload.get("dataset_path")
    if not dataset_path:
        return None
    return Path(dataset_path).expanduser().resolve()


def _flatten_context_entries(entries: list) -> list[str]:
    docs = []
    for entry in entries:
        if isinstance(entry, str) and entry.strip():
            docs.append(entry.strip()[:2000])
            continue
        if not isinstance(entry, dict):
            continue
        parts = [
            entry.get("title", ""),
            entry.get("subtitle", ""),
            entry.get("description", ""),
            entry.get("text", ""),
            entry.get("content", ""),
            " ".join(v for v in entry.get("ko_content_flat", []) if isinstance(v, str)),
        ]
        merged = ". ".join([part for part in parts if part]).strip()
        if merged:
            docs.append(merged[:2000])
    return docs


def _normalize_reference_facts(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    text = str(value).strip()
    return [text] if text else []


def build_reference_lookup(dataset_path: Path) -> dict:
    tasks, _ = load_vision_dataset(dataset_path)
    lookup = {}
    for task in tasks:
        key = (task["item_id"], task["question_id"], task["language"])
        context_documents = _flatten_context_entries(task.get("context") or [])
        context_documents.extend(_normalize_reference_facts(task.get("reference_texts")))
        source_text = str(task.get("source_text") or "").strip()
        if source_text:
            context_documents.append(source_text[:4000])
        lookup[key] = {
            "task_type": task.get("task_type", "qa"),
            "modality": task.get("modality", "image"),
            "reference_facts": _normalize_reference_facts(task.get("reference_facts")),
            "context_documents": context_documents,
            "expected_elements": task.get("expected_elements") or [],
            "source_text": source_text,
            "max_sentences": int(task.get("max_sentences", 8)),
        }
    return lookup


def main():
    base_results_dir = (PROJECT_ROOT / "results").resolve()
    resolved = resolve_scoring_paths(base_results_dir)
    db_path = resolved["db_path"]
    scores_db_path = resolved["scores_db_path"]
    scores_xlsx_path = resolved["scores_xlsx_path"]

    print("=" * 60)
    print("  Evaluating Multimodal Responses")
    print(f"  Database: {db_path}")
    print("=" * 60)
    print(f"  Path mode: {resolved['source']}")
    print(f"  GPU bucket: {resolved['gpu_bucket']} ({resolved['gpu_source']})")
    if resolved["run_dir"]:
        print(f"  Run dir: {resolved['run_dir']}")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("   Run multimodal evaluation first: python gpu_runtime/evaluate_vision.py")
        sys.exit(1)

    dataset_env = os.getenv("EVAL_VISION_DATASET", "").strip()
    dataset_path = Path(dataset_env).expanduser().resolve() if dataset_env else None
    if not dataset_path and resolved["run_dir"]:
        dataset_path = _read_run_dataset_path(resolved["run_dir"])
    if not dataset_path:
        dataset_path = (PROJECT_ROOT / "data" / "evaluation_vision_questions.json").resolve()
    if not dataset_path.exists():
        print(f"❌ Dataset not found for scoring: {dataset_path}")
        sys.exit(2)

    print(f"\n📚 Dataset: {dataset_path}")
    reference_lookup = build_reference_lookup(dataset_path)

    print("\n🔄 Loading evaluation models...")
    evaluator = ResponseEvaluator(metrics_profile="context")
    print("✅ Models loaded")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, model_name, language, question_id, item_id, task_type, modality,
               run_number, question_text, context, response
        FROM evaluations
        ORDER BY model_name, item_id, language, question_id, run_number
    """)
    evaluations = cursor.fetchall()
    conn.close()

    print(f"\n📂 Found {len(evaluations)} multimodal responses")

    scores_conn = sqlite3.connect(scores_db_path)
    scores_cursor = scores_conn.cursor()
    scores_cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            model_name TEXT,
            language TEXT,
            question_id TEXT,
            item_id TEXT,
            task_type TEXT,
            modality TEXT,
            relevance REAL,
            factual_accuracy REAL,
            completeness REAL,
            fluency REAL,
            coherence REAL,
            prompt_alignment REAL,
            token_efficiency REAL,
            overall_quality REAL,
            timestamp TEXT
        )
    """)
    scores_cursor.execute("DELETE FROM scores")
    scores_conn.commit()

    print("\n🔍 Evaluating responses...")
    prepared = []
    for row in evaluations:
        (
            eval_id, model_name, language, question_id, item_id, task_type, modality,
            _run_number, question_text, context_json, response,
        ) = row
        if not response:
            continue

        lookup_key = (item_id, question_id, language)
        ref_data = dict(reference_lookup.get(lookup_key, {}))
        if not ref_data:
            ref_data = {
                "task_type": task_type or "qa",
                "modality": modality or "image",
                "reference_facts": [],
                "context_documents": [],
                "expected_elements": [],
                "source_text": "",
                "max_sentences": 8,
            }
        try:
            raw_context = json.loads(context_json) if context_json else []
        except Exception:
            raw_context = []
        ref_data["context_documents"] = list(ref_data.get("context_documents", [])) + _flatten_context_entries(raw_context)
        prepared.append({
            "eval_id": eval_id,
            "model_name": model_name,
            "language": language,
            "question_id": question_id,
            "item_id": item_id,
            "task_type": task_type or ref_data.get("task_type", "qa"),
            "modality": modality or ref_data.get("modality", "image"),
            "question_text": question_text,
            "response": response,
            "ref_data": ref_data,
        })

    batch_size = max(1, int(os.getenv("EVALUATOR_SCORE_BATCH_SIZE", "96")))
    commit_every = max(1, int(os.getenv("EVALUATOR_SCORE_COMMIT_EVERY", "200")))
    inserted = 0

    pbar = tqdm(total=len(prepared))
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        responses = [item["response"] for item in batch]
        languages = [item["language"] for item in batch]
        contexts_batch = [item["ref_data"].get("context_documents", []) for item in batch]

        fluency_scores = None
        coherence_scores = None
        nli_scores = None
        try:
            fluency_scores = evaluator.calculate_fluency_batch(responses, languages)
            coherence_scores = evaluator.calculate_coherence_batch(responses)
            nli_scores = evaluator.calculate_nli_entailment_batch(responses, contexts_batch)
        except Exception as e:
            print(f"\n   ⚠️ Batch precompute fallback at offset {start}: {e}")

        for idx, item in enumerate(batch):
            try:
                precomputed = {}
                if fluency_scores is not None and idx < len(fluency_scores):
                    precomputed["fluency"] = float(fluency_scores[idx])
                if coherence_scores is not None and idx < len(coherence_scores):
                    precomputed["coherence"] = float(coherence_scores[idx])
                if nli_scores is not None and idx < len(nli_scores):
                    precomputed["factual_accuracy"] = float(nli_scores[idx])

                metric_question_id = item["question_id"]
                if item["task_type"] == "summary":
                    metric_question_id = "Q5_SUMMARIZATION_ACCURACY"

                ref_data = {
                    "reference_facts": item["ref_data"].get("reference_facts", []),
                    "context_documents": item["ref_data"].get("context_documents", []),
                    "expected_elements": item["ref_data"].get("expected_elements", []),
                    "source_text": item["ref_data"].get("source_text", ""),
                    "max_sentences": item["ref_data"].get("max_sentences", 8),
                }

                scores = evaluator.evaluate_response(
                    question_id=metric_question_id,
                    question_text=item["question_text"],
                    response_text=item["response"],
                    language=item["language"],
                    tokens_generated=len(item["response"].split()),
                    reference_data=ref_data,
                    precomputed_scores=precomputed if precomputed else None,
                )

                scores_cursor.execute("""
                    INSERT INTO scores
                    (evaluation_id, model_name, language, question_id, item_id, task_type, modality,
                     relevance, factual_accuracy, completeness, fluency, coherence, prompt_alignment,
                     token_efficiency, overall_quality, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    item["eval_id"],
                    item["model_name"],
                    item["language"],
                    item["question_id"],
                    item["item_id"],
                    item["task_type"],
                    item["modality"],
                    scores.relevance,
                    scores.factual_accuracy,
                    scores.completeness,
                    scores.fluency,
                    scores.coherence,
                    scores.prompt_alignment,
                    scores.token_efficiency,
                    scores.overall_quality,
                ))
                inserted += 1
                if inserted % commit_every == 0:
                    scores_conn.commit()
            except Exception as e:
                print(f"\n   ⚠️ Error evaluating {item['item_id']}:{item['question_id']} ({item['language']}): {e}")
                continue

        pbar.update(len(batch))
    pbar.close()

    scores_conn.commit()
    scores_conn.close()

    if resolved["run_dir"]:
        meta = {
            "updated_at": datetime.now().isoformat(),
            "dataset_path": str(dataset_path),
            "scoring_db": str(scores_db_path),
            "scoring_xlsx": str(scores_xlsx_path),
            "source_results_db": str(db_path),
            "gpu_bucket": resolved["gpu_bucket"],
            "gpu_detected_from": resolved["gpu_source"],
            "path_mode": resolved["source"],
        }
        with open(resolved["metadata_dir"] / "scoring_info.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n💾 Scores saved to: {scores_db_path}")
    print("\n📊 Summary:")

    summary_conn = sqlite3.connect(scores_db_path)
    summary_cursor = summary_conn.cursor()
    summary_cursor.execute("""
        SELECT model_name, COUNT(*), AVG(overall_quality)
        FROM scores
        GROUP BY model_name
        ORDER BY AVG(overall_quality) DESC
    """)
    for row in summary_cursor.fetchall():
        print(f"   {row[0]}: {row[1]} responses, avg quality: {row[2]:.3f}")
    summary_conn.close()

    if export_scores_to_excel(scores_db_path, scores_xlsx_path):
        print(f"📄 Excel exported to: {scores_xlsx_path}")


if __name__ == "__main__":
    main()
