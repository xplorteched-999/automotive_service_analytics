from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "sql_outputs"


print("\n" + "=" * 70)
print("SQL ANALYSIS RESULT SUMMARY")
print("=" * 70)


files = sorted(
    OUTPUT_DIR.glob("*.csv")
)


if not files:

    print(
        "\nNo SQL output files found."
    )

    print(
        "Run: python src/run_sql_analysis.py"
    )

    raise SystemExit


for file in files:

    print("\n" + "-" * 70)

    print(file.name)

    print("-" * 70)

    df = pd.read_csv(file)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nColumns:"
    )

    print(
        ", ".join(df.columns)
    )

    print(
        "\nPreview:"
    )

    print(
        df.head(5).to_string(index=False)
    )


print("\n" + "=" * 70)
print("SUMMARY COMPLETE")
print("=" * 70)