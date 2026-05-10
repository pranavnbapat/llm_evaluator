# gpu_runtime/evaluate_context_results.py

#!/usr/bin/env python3
"""
Evaluate context-based responses from SQLite database using scientific metrics.

Usage:
    python gpu_runtime/evaluate_context_results.py

This will:
    1. Read all responses from results/evaluation_results_euf_context.db
    2. Compute quality scores for each response
    3. Save scores to results/evaluation_scores_euf_context.db
"""
import sqlite3
import sys
import os
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics.scientific_metrics import ResponseEvaluator
from translations.eu_24_languages_euf_context import get_all_questions_with_context


def detect_gpu_bucket() -> tuple[str, str]:
    """Detect GPU bucket with optional env override."""
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
    """
    Resolve where to read evaluation DB and write scoring outputs.
    Priority:
      1) EVAL_RUN_DIR
      2) EVAL_RUN_GPU + EVAL_RUN_ID
      3) results/latest/<gpu_bucket>
      4) legacy results/ directory
    """
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
        latest_link = (base_results_dir / "latest" / gpu_bucket)
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
        db_path = raw_dir / "evaluation_results_euf_context.db"
        scores_db_path = scores_dir / "evaluation_scores_euf_context.db"
        scores_xlsx_path = scores_dir / "evaluation_scores_euf_context.xlsx"
    else:
        db_path = base_results_dir / "evaluation_results_euf_context.db"
        scores_db_path = base_results_dir / "evaluation_scores_euf_context.db"
        scores_xlsx_path = base_results_dir / "evaluation_scores_euf_context.xlsx"
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


def load_env_file(env_path: Path) -> dict:
    """Lightweight .env parser (KEY=VALUE)."""
    data = {}
    if not env_path.exists():
        return data
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def getenv_int(name: str, default: int, env_file_vals: dict) -> int:
    """Read int from shell env first, then .env map, then default."""
    raw = os.getenv(name)
    if raw is None:
        raw = env_file_vals.get(name)
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def export_scores_to_excel(scores_db_path: Path, out_xlsx_path: Path) -> bool:
    """Export scores table to Excel. Returns True on success."""
    try:
        import pandas as pd
    except ImportError:
        print("\n⚠️ Skipping Excel export: pandas/openpyxl not installed.")
        print("   Install with: uv pip install --python .venv/bin/python pandas openpyxl")
        return False

    conn = sqlite3.connect(scores_db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM scores", conn)
    finally:
        conn.close()

    with pd.ExcelWriter(out_xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="scores", index=False)
    return True


def get_context_for_question(question_id: str) -> list:
    """Get context (search results) for a question."""
    questions = get_all_questions_with_context()
    
    for q in questions:
        if q['question_id'] == question_id:
            return q.get('context', [])
    
    return []


def main():
    """Main evaluation function."""
    base_results_dir = (PROJECT_ROOT / "results").resolve()
    resolved = resolve_scoring_paths(base_results_dir)
    db_path = resolved["db_path"]
    scores_db_path = resolved["scores_db_path"]
    scores_xlsx_path = resolved["scores_xlsx_path"]

    print("="*60)
    print("  Evaluating Context-Based Responses")
    print(f"  Database: {db_path}")
    print("="*60)

    print(f"  Path mode: {resolved['source']}")
    print(f"  GPU bucket: {resolved['gpu_bucket']} ({resolved['gpu_source']})")
    if resolved["run_dir"]:
        print(f"  Run dir: {resolved['run_dir']}")
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Run evaluation first: python gpu_runtime/evaluate_context.py")
        sys.exit(1)
    
    # Initialize evaluator
    print("\n🔄 Loading evaluation models...")
    evaluator = ResponseEvaluator(metrics_profile="context")
    print("✅ Models loaded")
    
    # Connect to database
    print(f"\n📂 Reading responses from: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all evaluations
    cursor.execute("""
        SELECT id, model_name, language, question_id, run_number, 
               question_text, context, response, latency_ms
        FROM evaluations
        ORDER BY model_name, language, question_id, run_number
    """)
    
    evaluations = cursor.fetchall()
    conn.close()
    
    print(f"📊 Found {len(evaluations)} responses to evaluate")
    
    # Setup scores database
    scores_conn = sqlite3.connect(scores_db_path)
    scores_cursor = scores_conn.cursor()
    
    scores_cursor.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            model_name TEXT,
            language TEXT,
            question_id TEXT,
            run_number INTEGER,
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
    # Older databases may pre-date the run_number column; add it if absent so the
    # UNIQUE index below can be created without losing existing rows.
    cols = {row[1] for row in scores_cursor.execute("PRAGMA table_info(scores)").fetchall()}
    if "run_number" not in cols:
        scores_cursor.execute("ALTER TABLE scores ADD COLUMN run_number INTEGER")
    scores_cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uniq_scores_cell "
        "ON scores(model_name, language, question_id, run_number)"
    )
    scores_conn.commit()
    # Resumable scoring: keep prior rows. Each (model, lang, qid, run_number)
    # cell is upserted via INSERT OR REPLACE on the unique index above. Pass
    # FORCE_RESCORE=1 to wipe and re-score everything.
    if str(os.getenv("FORCE_RESCORE", "")).strip() in {"1", "true", "yes"}:
        scores_cursor.execute("DELETE FROM scores")
        scores_conn.commit()
        print("⚠️  FORCE_RESCORE=1 set; cleared existing scores rows.")

    # Evaluate each response (batched inference for heavy metrics)
    print("\n🔍 Evaluating responses...")
    prepared = []
    for eval_data in evaluations:
        (eval_id, model_name, language, question_id, run_number,
         question_text, context_json, response, latency_ms) = eval_data
        if not response:
            continue
        try:
            context = json.loads(context_json) if context_json else []
        except Exception:
            context = []
        context_documents = []
        for ctx in context:
            if isinstance(ctx, dict):
                title = ctx.get("title", "")
                subtitle = ctx.get("subtitle", "")
                description = ctx.get("description", "")
                keywords = ctx.get("keywords", [])
                ko_flat = ctx.get("ko_content_flat", [])
                keywords_text = ", ".join([k for k in keywords if isinstance(k, str)])
                ko_text = " ".join([k for k in ko_flat if isinstance(k, str)])
                parts = [title, subtitle, description, keywords_text, ko_text]
                full_text = ". ".join([p for p in parts if p]).strip()
                if full_text:
                    context_documents.append(full_text[:2000])
        ref_data = {
            "reference_facts": context_documents.copy(),
            "context_documents": context_documents,
        }
        prepared.append({
            "eval_id": eval_id,
            "model_name": model_name,
            "language": language,
            "question_id": question_id,
            "run_number": run_number,
            "question_text": question_text,
            "response": response,
            "ref_data": ref_data,
        })
    
    env_file_vals = load_env_file(PROJECT_ROOT / ".env")
    batch_size = max(1, getenv_int("EVALUATOR_SCORE_BATCH_SIZE", 96, env_file_vals))
    commit_every = max(1, getenv_int("EVALUATOR_SCORE_COMMIT_EVERY", 200, env_file_vals))
    inserted = 0
    pbar = tqdm(total=len(prepared))
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        responses = [b["response"] for b in batch]
        languages = [b["language"] for b in batch]
        contexts_batch = [b["ref_data"].get("context_documents", []) for b in batch]
        
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
                
                # Char-estimated token count, consistent with the budget CSV's
                # `char_estimated` path (insights/generate_context_token_budget.py).
                response_text = item["response"] or ""
                tokens_generated = max(1, int(round(len(response_text) / 4.0)))

                scores = evaluator.evaluate_response(
                    question_id=item["question_id"],
                    question_text=item["question_text"],
                    response_text=response_text,
                    language=item["language"],
                    tokens_generated=tokens_generated,
                    reference_data=item["ref_data"],
                    precomputed_scores=precomputed if precomputed else None,
                )

                scores_cursor.execute("""
                    INSERT OR REPLACE INTO scores
                    (evaluation_id, model_name, language, question_id, run_number,
                     relevance, factual_accuracy, completeness, fluency, coherence,
                     prompt_alignment, token_efficiency, overall_quality, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    item["eval_id"], item["model_name"], item["language"], item["question_id"],
                    item["run_number"],
                    scores.relevance,
                    scores.factual_accuracy,
                    scores.completeness,
                    scores.fluency,
                    scores.coherence,
                    scores.prompt_alignment,
                    scores.token_efficiency,
                    scores.overall_quality
                ))
                inserted += 1
                if inserted % commit_every == 0:
                    scores_conn.commit()
            except Exception as e:
                print(f"\n   ⚠️ Error evaluating {item['question_id']} ({item['language']}): {e}")
                continue
        pbar.update(len(batch))
    pbar.close()
    
    scores_conn.commit()
    scores_conn.close()

    if resolved["run_dir"]:
        meta = {
            "updated_at": datetime.now().isoformat(),
            "scoring_db": str(scores_db_path),
            "scoring_xlsx": str(scores_xlsx_path),
            "source_results_db": str(db_path),
            "gpu_bucket": resolved["gpu_bucket"],
            "gpu_detected_from": resolved["gpu_source"],
            "path_mode": resolved["source"],
        }
        with open(resolved["metadata_dir"] / "scoring_info.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\n💾 Scores saved to: {scores_db_path}")
    print(f"\n📊 Summary:")
    
    # Show summary stats
    summary_conn = sqlite3.connect(scores_db_path)
    summary_cursor = summary_conn.cursor()
    
    summary_cursor.execute("""
        SELECT model_name, COUNT(*), AVG(overall_quality)
        FROM scores
        GROUP BY model_name
    """)
    
    for row in summary_cursor.fetchall():
        print(f"   {row[0]}: {row[1]} responses, avg quality: {row[2]:.3f}")
    
    summary_conn.close()

    # Export XLSX
    exported = export_scores_to_excel(scores_db_path, scores_xlsx_path)
    if exported:
        print(f"📄 Excel exported to: {scores_xlsx_path}")
    
    print("\nNext steps:")
    print("   1. Generate report: python generate_report.py")


if __name__ == "__main__":
    main()
