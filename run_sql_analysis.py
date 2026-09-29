from pathlib import Path
import duckdb
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
SQL_DIR = PROJECT_ROOT / "sql"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "sql_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROJECT_ROOT / "automotive_service.duckdb"


# ============================================================
# TABLE DEFINITIONS
# ============================================================

TABLES = [
    "customers",
    "vehicles",
    "dealers",
    "appointments",
    "service_visits",
    "repairs",
    "parts",
    "customer_surveys",
]


# ============================================================
# CREATE DUCKDB CONNECTION
# ============================================================

print("\nConnecting to DuckDB...")

con = duckdb.connect(str(DB_PATH))


# ============================================================
# LOAD PROCESSED CSV FILES
# ============================================================

print("\nLoading processed datasets...")

for table in TABLES:

    file_path = DATA_DIR / f"{table}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing processed file: {file_path}"
        )

    print(f"Loading {table}...")

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT *
        FROM read_csv_auto('{file_path.as_posix()}',
                           header=True,
                           ignore_errors=True)
        """
    )


# ============================================================
# DISPLAY ROW COUNTS
# ============================================================

print("\nDataset row counts:")

for table in TABLES:

    count = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f"{table:20s}: {count:,}")


# ============================================================
# SQL FILE EXECUTION FUNCTION
# ============================================================

# ============================================================
# SQL FILE EXECUTION FUNCTION
# ============================================================

def execute_sql_file(sql_file):

    print("\n" + "=" * 70)
    print(f"Executing: {sql_file.name}")
    print("=" * 70)

    sql_text = sql_file.read_text(encoding="utf-8")

    # Split SQL file into individual statements
    statements = [
        statement.strip()
        for statement in sql_text.split(";")
        if statement.strip()
    ]

    result_counter = 0

    for statement in statements:

        # Remove SQL comment lines before checking
        # whether the statement is a SELECT/WITH query.
        cleaned_lines = []

        for line in statement.splitlines():

            stripped_line = line.strip()

            if stripped_line.startswith("--"):
                continue

            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines).strip()

        # Skip empty/comment-only statements
        if not cleaned:
            continue

        try:

            result = con.execute(statement)

            # Export SELECT / WITH query results
            if cleaned.upper().startswith(("SELECT", "WITH")):

                df = result.df()

                result_counter += 1

                output_name = (
                    f"{sql_file.stem}_{result_counter:02d}.csv"
                )

                output_path = OUTPUT_DIR / output_name

                df.to_csv(
                    output_path,
                    index=False
                )

                print(
                    f"Saved {output_name} "
                    f"({len(df):,} rows)"
                )

        except Exception as e:

            print(
                f"\nERROR in {sql_file.name}"
            )

            print(
                f"\nQuery:\n{cleaned[:1000]}"
            )

            print(
                f"\nError: {e}"
            )

            raise


# ============================================================
# EXECUTE PHASE 4 SQL FILES
# ============================================================

SQL_FILES = [
    SQL_DIR / "01_kpi_analysis.sql",
    SQL_DIR / "02_root_cause_analysis.sql",
    SQL_DIR / "03_customer_segmentation.sql",
    SQL_DIR / "04_advanced_analysis.sql",
]


for sql_file in SQL_FILES:

    if not sql_file.exists():

        print(
            f"\nSkipping missing SQL file: {sql_file}"
        )

        continue

    execute_sql_file(sql_file)


# ============================================================
# CLOSE DATABASE
# ============================================================

con.close()

print("\n" + "=" * 70)
print("PHASE 4 SQL ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nOutputs saved to:\n{OUTPUT_DIR}"
)

print(
    f"\nDuckDB database:\n{DB_PATH}"
)