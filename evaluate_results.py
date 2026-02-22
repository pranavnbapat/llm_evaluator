#!/usr/bin/env python3
"""
Evaluate all responses from SQLite database using scientific metrics.

Usage:
    python evaluate_results.py

This will:
    1. Read all responses from results/evaluation_results.db
    2. Compute quality scores for each response
    3. Save scores to results/evaluation_scores.db
"""
import sqlite3
import sys
from pathlib import Path
# from tqdm import tqdm  # Optional progress bar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from metrics.scientific_metrics import ResponseEvaluator, QualityScores
from translations.eu_24_languages import get_all_questions, get_question_metadata


def get_reference_data(question_id: str) -> dict:
    """Get reference data for a question."""
    metadata = get_question_metadata()
    
    q_map = {
        "Q1_FACTUAL_KNOWLEDGE": "Q1",
        "Q2_LOGICAL_REASONING": "Q2",
        "Q3_INSTRUCTION_FOLLOWING": "Q3",
        "Q4_CULTURAL_NUANCE": "Q4",
        "Q5_SUMMARIZATION_ACCURACY": "Q5",
    }
    
    meta_key = q_map.get(question_id)
    if not meta_key or meta_key not in metadata:
        return {}
    
    meta = metadata[meta_key]
    
    ref_data = {
        "expected_elements": meta.get("expected_elements", []),
        "required_keys": meta.get("required_keys", []),
        "max_sentences": meta.get("max_sentences", 3),
    }
    
    # Add question-specific reference facts
    if question_id == "Q1_FACTUAL_KNOWLEDGE":
        ref_data["reference_facts"] = {
            "capital": "Lisbon",
            "population": "500000",
            "eu_membership": "1986",
            "location": "Iberian Peninsula",
        }
    elif question_id == "Q2_LOGICAL_REASONING":
        ref_data["expected_answer"] = meta.get("expected_answer")
    
    return ref_data


def main():
    """Main evaluation function."""
    print("="*60)
    print("  Evaluating Responses with Scientific Metrics")
    print("="*60)
    
    # Paths
    results_dir = Path("results")
    db_path = results_dir / "evaluation_results.db"
    scores_db_path = results_dir / "evaluation_scores.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)
    
    # Initialize evaluator
    print("\n🔄 Loading evaluation models...")
    evaluator = ResponseEvaluator()
    
    # Load questions
    all_questions = get_all_questions()
    
    # Connect to databases
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create scores database
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
            FOREIGN KEY (evaluation_id) REFERENCES evaluations(id)
        )
    """)
    scores_conn.commit()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM evaluations")
    total = cursor.fetchone()[0]
    
    print(f"\n📊 Total responses to evaluate: {total}")
    print("="*60)
    
    # Process each response
    cursor.execute("""
        SELECT id, model_name, language, question_id, response 
        FROM evaluations
    """)
    
    rows = cursor.fetchall()
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        eval_id, model_name, language, question_id, response = row
        
        # Get question text
        question_text = all_questions.get(language, {}).get(question_id, "")
        
        # Get reference data
        reference_data = get_reference_data(question_id)
        
        # Evaluate
        try:
            scores = evaluator.evaluate_response(
                question_id=question_id,
                question_text=question_text,
                response_text=response,
                language=language.lower(),
                tokens_generated=len(response.split()),
                reference_data=reference_data,
            )
            
            # Save scores
            scores_cursor.execute("""
                INSERT INTO scores (
                    evaluation_id, model_name, language, question_id,
                    relevance, factual_accuracy, completeness, fluency,
                    coherence, prompt_alignment, token_efficiency, overall_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id, model_name, language, question_id,
                scores.relevance, scores.factual_accuracy, scores.completeness,
                scores.fluency, scores.coherence, scores.prompt_alignment,
                scores.token_efficiency, scores.overall_quality
            ))
            
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}/{language}/{question_id}: {e}")
            continue
    
    scores_conn.commit()
    conn.close()
    scores_conn.close()
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print(f"Scores saved to: {scores_db_path}")
    print("="*60)


if __name__ == "__main__":
    main()
