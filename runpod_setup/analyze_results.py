#!/usr/bin/env python3
"""
Analyze evaluation results downloaded from RunPod.

Usage:
    python analyze_results.py ./runpod_results_20240220_143022/
"""
import json
import sqlite3
import sys
from pathlib import Path
import pandas as pd


def analyze_json_results(results_dir: Path):
    """Analyze JSON result files."""
    print("=" * 80)
    print("JSON RESULTS SUMMARY")
    print("=" * 80)
    
    json_files = list(results_dir.glob("*_results.json"))
    
    if not json_files:
        print("No JSON result files found")
        return
    
    summaries = []
    
    for json_file in sorted(json_files):
        with open(json_file) as f:
            data = json.load(f)
        
        model_name = data.get("model_name", json_file.stem)
        avg_score = data.get("average_quality_score", 0)
        successful = data.get("successful", 0)
        total = data.get("total_runs", 0)
        
        summaries.append({
            "model": model_name,
            "avg_score": avg_score,
            "successful": successful,
            "total": total,
        })
        
        print(f"\n📊 {model_name}")
        print(f"   Average Quality Score: {avg_score:.4f}")
        print(f"   Successful Runs: {successful}/{total}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    df = pd.DataFrame(summaries)
    df = df.sort_values("avg_score", ascending=False)
    print(df.to_string(index=False))


def analyze_sqlite(db_path: Path):
    """Analyze SQLite database."""
    print("\n" + "=" * 80)
    print("SQLITE DATABASE SUMMARY")
    print("=" * 80)
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Get all tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\nTables: {[t[0] for t in tables]}")
    
    # Try to get evaluation summary
    try:
        df = pd.read_sql("""
            SELECT 
                model_name,
                AVG(quality_score) as avg_score,
                COUNT(*) as num_runs,
                AVG(latency_ms) as avg_latency
            FROM evaluations
            GROUP BY model_name
            ORDER BY avg_score DESC
        """, conn)
        
        print("\n📊 Per-Model Summary:")
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"Could not query evaluations table: {e}")
    
    conn.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)
    
    # Analyze JSON files
    analyze_json_results(results_dir)
    
    # Analyze SQLite if present
    db_path = results_dir / "evaluation_results.db"
    analyze_sqlite(db_path)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
