# evaluate_context_results.py

#!/usr/bin/env python3
"""
Evaluate context-based responses from SQLite database using scientific metrics.

Usage:
    python evaluate_context_results.py

This will:
    1. Read all responses from results/evaluation_results_euf_context.db
    2. Compute quality scores for each response
    3. Save scores to results/evaluation_scores_euf_context.db
"""
import sqlite3
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from metrics.scientific_metrics import ResponseEvaluator
from translations.eu_24_languages_euf_context import get_all_questions_with_context


def export_scores_to_excel(scores_db_path: Path, out_xlsx_path: Path) -> bool:
    """Export scores table to Excel. Returns True on success."""
    try:
        import pandas as pd
    except ImportError:
        print("\n⚠️ Skipping Excel export: pandas/openpyxl not installed.")
        print("   Install with: pip install pandas openpyxl")
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
    print("="*60)
    print("  Evaluating Context-Based Responses")
    print("  Database: evaluation_results_euf_context.db")
    print("="*60)
    
    # Paths
    results_dir = Path("results")
    db_path = results_dir / "evaluation_results_euf_context.db"
    scores_db_path = results_dir / "evaluation_scores_euf_context.db"
    scores_xlsx_path = results_dir / "evaluation_scores_euf_context.xlsx"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print(f"   Run evaluation first: python runpod_setup/evaluate_context.py")
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
    
    # Clear existing scores to avoid duplicates
    scores_cursor.execute("DELETE FROM scores")
    scores_conn.commit()
    
    # Evaluate each response
    print("\n🔍 Evaluating responses...")
    
    for eval_data in tqdm(evaluations):
        (eval_id, model_name, language, question_id, run_number,
         question_text, context_json, response, latency_ms) = eval_data
        
        if not response:
            continue
        
        # Parse context from JSON
        import json
        try:
            context = json.loads(context_json) if context_json else []
        except:
            context = []
        
        # Extract full context documents for semantic comparison
        context_documents = []
        for ctx in context:
            if isinstance(ctx, dict):
                # Combine available fields for full context
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
                    context_documents.append(full_text[:2000])  # First 2000 chars
        
        # Also extract reference facts as fallback
        reference_facts = context_documents.copy()
        
        # Build reference data
        ref_data = {
            "reference_facts": reference_facts,
            "context_documents": context_documents,
        }
        
        # Evaluate response
        try:
            scores = evaluator.evaluate_response(
                question_id=question_id,
                question_text=question_text,
                response_text=response,
                language=language,
                tokens_generated=len(response.split()),  # Approximate
                reference_data=ref_data
            )
            
            # Save scores
            scores_cursor.execute("""
                INSERT INTO scores 
                (evaluation_id, model_name, language, question_id, relevance, 
                 factual_accuracy, completeness, fluency, coherence, prompt_alignment,
                 token_efficiency, overall_quality, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                eval_id, model_name, language, question_id,
                scores.relevance,
                scores.factual_accuracy,
                scores.completeness,
                scores.fluency,
                scores.coherence,
                scores.prompt_alignment,
                scores.token_efficiency,
                scores.overall_quality
            ))
            
        except Exception as e:
            print(f"\n   ⚠️ Error evaluating {question_id} ({language}): {e}")
            continue
    
    scores_conn.commit()
    scores_conn.close()
    
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
