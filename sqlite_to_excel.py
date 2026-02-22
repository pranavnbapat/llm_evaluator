# sqlite_to_excel.py

import sqlite3
from pathlib import Path

import pandas as pd


# ---------- CONFIG ----------
SQLITE_PATH = Path("results/evaluation_results.db")
OUT_XLSX = Path("evaluation_results.xlsx")
# ---------------------------------------------


def get_user_tables(con: sqlite3.Connection) -> list[str]:
    """
    Returns non-internal SQLite tables (ignores sqlite_sequence etc).
    """
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
    """
    rows = con.execute(q).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"SQLite file not found: {SQLITE_PATH}")

    con = sqlite3.connect(str(SQLITE_PATH))

    try:
        tables = get_user_tables(con)

        if len(tables) == 0:
            raise RuntimeError("No user tables found in this SQLite database.")
        if len(tables) > 1:
            raise RuntimeError(
                f"Expected 1 user table, found {len(tables)}: {tables}\n"
                "If you actually want a specific table, set TABLE_NAME explicitly."
            )

        table = tables[0]

        # Read everything from the single table
        df = pd.read_sql_query(f'SELECT * FROM "{table}";', con)

        # Excel sheet name limits: 31 chars
        sheet_name = table[:31] if table else "Sheet1"

        # Write to XLSX
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Exported table '{table}' ({len(df)} rows, {len(df.columns)} cols) -> {OUT_XLSX.resolve()}")

    finally:
        con.close()


if __name__ == "__main__":
    main()