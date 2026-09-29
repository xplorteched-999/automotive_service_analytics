import os
import pandas as pd
import numpy as np
import matplotlib

# Use a non-GUI backend so charts save correctly on Windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase5"
CHART_PATH = f"{REPORT_PATH}/charts"

os.makedirs(REPORT_PATH, exist_ok=True)
os.makedirs(CHART_PATH, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading data...")

visits = pd.read_csv(
    f"{DATA_PATH}/service_visits.csv"
)

appointments = pd.read_csv(
    f"{DATA_PATH}/appointments.csv"
)

surveys = pd.read_csv(
    f"{DATA_PATH}/customer_surveys.csv"
)
print(appointments.columns.tolist())

repairs = pd.read_csv(
    f"{DATA_PATH}/repairs.csv"
)

parts = pd.read_csv(
    f"{DATA_PATH}/parts.csv"
)

print("Visits:", visits.shape)
print("Appointments:", appointments.shape)
print("Surveys:", surveys.shape)
print("Repairs:", repairs.shape)
print("Parts:", parts.shape)


# ============================================================
# 3. BUILD MAIN ANALYSIS DATASET
# ============================================================

print("\nBuilding analysis dataset...")

df = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "vehicle_id",
            "appointment_date",
            "booking_date",
        ]
    ],
    on="appointment_id",
    how="left"
)


df = df.merge(
    surveys[
        [
            "visit_id",
            "csat_score",
            "complaint_flag"
        ]
    ],
    on="visit_id",
    how="left"
)


# ============================================================
# 4. DATE / FEATURE ENGINEERING
# ============================================================

df["appointment_date"] = pd.to_datetime(
    df["appointment_date"],
    errors="coerce"
)

df["booking_date"] = pd.to_datetime(
    df["booking_date"],
    errors="coerce"
)

df["lead_time_days"] = (
    df["appointment_date"] -
    df["booking_date"]
).dt.days


# Delay status
df["delayed"] = (
    df["delay_hours"] > 0
)


# Delay categories
df["delay_category"] = np.select(
    [
        df["delay_hours"] <= 0,
        df["delay_hours"] <= 4,
        df["delay_hours"] <= 8,
        df["delay_hours"] > 8
    ],
    [
        "On Time",
        "Minor Delay",
        "Moderate Delay",
        "Severe Delay"
    ],
    default="Unknown"
)

print("Analysis dataset:", df.shape)


# ============================================================
# 5. OVERALL KPI SUMMARY
# ============================================================

print("\nCalculating overall KPIs...")

total_visits = len(df)

delayed_visits = df["delayed"].sum()

delay_rate = (
    delayed_visits / total_visits
)

avg_delay = df["delay_hours"].mean()

median_delay = df["delay_hours"].median()

avg_csat = df["csat_score"].mean()

complaint_rate = df["complaint_flag"].mean()


overall_summary = pd.DataFrame({
    "metric": [
        "Total Visits",
        "Delayed Visits",
        "Delay Rate",
        "Average Delay Hours",
        "Median Delay Hours",
        "Average CSAT",
        "Complaint Rate"
    ],
    "value": [
        total_visits,
        delayed_visits,
        delay_rate,
        avg_delay,
        median_delay,
        avg_csat,
        complaint_rate
    ]
})

overall_summary.to_csv(
    f"{REPORT_PATH}/overall_summary.csv",
    index=False
)

print("\nOverall Summary")
print(overall_summary)


# ============================================================
# 6. DELAY DISTRIBUTION
# ============================================================

delay_distribution = (
    df["delay_category"]
    .value_counts()
    .reindex(
        [
            "On Time",
            "Minor Delay",
            "Moderate Delay",
            "Severe Delay"
        ],
        fill_value=0
    )
    .reset_index()
)

delay_distribution.columns = [
    "delay_category",
    "visits"
]

delay_distribution["percentage"] = (
    delay_distribution["visits"] /
    total_visits *
    100
)

delay_distribution.to_csv(
    f"{REPORT_PATH}/delay_distribution.csv",
    index=False
)


# Chart 1
plt.figure(figsize=(8, 5))

plt.bar(
    delay_distribution["delay_category"],
    delay_distribution["visits"]
)

plt.title("Service Visit Delay Distribution")
plt.xlabel("Delay Category")
plt.ylabel("Number of Visits")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    f"{CHART_PATH}/01_delay_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 7. QUESTION 1 — DELAY VS CSAT
# ============================================================

print("\n----------------------------------------")
print("TEST 1 — SERVICE DELAY VS CSAT")
print("----------------------------------------")

on_time_csat = df.loc[
    ~df["delayed"],
    "csat_score"
].dropna()

delayed_csat = df.loc[
    df["delayed"],
    "csat_score"
].dropna()


u_stat, p_value = mannwhitneyu(
    on_time_csat,
    delayed_csat,
    alternative="two-sided"
)


test_1 = pd.DataFrame({
    "test": ["Mann-Whitney U"],
    "question": [
        "Is CSAT different between delayed and on-time visits?"
    ],
    "on_time_median_csat": [
        on_time_csat.median()
    ],
    "delayed_median_csat": [
        delayed_csat.median()
    ],
    "u_statistic": [
        u_stat
    ],
    "p_value": [
        p_value
    ],
    "significant_at_0_05": [
        p_value < 0.05
    ]
})

test_1.to_csv(
    f"{REPORT_PATH}/test_01_csat_delay.csv",
    index=False
)

print(test_1.to_string(index=False))


# ============================================================
# 8. CSAT CHART
# ============================================================

plot_df = df[
    ["csat_score", "delayed"]
].dropna().copy()

plot_df["visit_status"] = np.where(
    plot_df["delayed"],
    "Delayed",
    "On Time"
)

csat_on_time = plot_df.loc[
    plot_df["visit_status"] == "On Time",
    "csat_score"
]

csat_delayed = plot_df.loc[
    plot_df["visit_status"] == "Delayed",
    "csat_score"
]

plt.figure(figsize=(7, 5))

plt.boxplot(
    [
        csat_on_time,
        csat_delayed
    ],
    tick_labels=[
        "On Time",
        "Delayed"
    ]
)


plt.title("Customer Satisfaction: Delayed vs On-Time Visits")
plt.xlabel("Visit Status")
plt.ylabel("CSAT Score")

plt.tight_layout()

plt.savefig(
    f"{CHART_PATH}/02_csat_delayed_vs_ontime.png",
    dpi=150
)

plt.close()


# ============================================================
# 9. QUESTION 2 — DELAY VS COMPLAINT
# ============================================================

print("\n----------------------------------------")
print("TEST 2 — SERVICE DELAY VS COMPLAINT")
print("----------------------------------------")


complaint_data = df[
    ["delayed", "complaint_flag"]
].dropna()


complaint_table = pd.crosstab(
    complaint_data["delayed"],
    complaint_data["complaint_flag"]
)


chi2, p_value, degrees_of_freedom, expected = (
    chi2_contingency(complaint_table)
)


test_2 = pd.DataFrame({
    "test": ["Chi-Square Test"],
    "question": [
        "Is service delay associated with customer complaints?"
    ],
    "chi_square": [
        chi2
    ],
    "degrees_of_freedom": [
        degrees_of_freedom
    ],
    "p_value": [
        p_value
    ],
    "significant_at_0_05": [
        p_value < 0.05
    ]
})

test_2.to_csv(
    f"{REPORT_PATH}/test_02_delay_complaint.csv",
    index=False
)

print(test_2.to_string(index=False))


# ============================================================
# 10. COMPLAINT RATE BY DELAY STATUS
# ============================================================

complaint_summary = (
    df.groupby("delayed")["complaint_flag"]
    .agg(
        complaint_rate="mean",
        visits="count"
    )
    .reset_index()
)

complaint_summary["visit_status"] = np.where(
    complaint_summary["delayed"],
    "Delayed",
    "On Time"
)

complaint_summary.to_csv(
    f"{REPORT_PATH}/complaint_by_delay_status.csv",
    index=False
)


# ============================================================
# 11. PARTS DATA — IDENTIFY DATE COLUMNS
# ============================================================

print("\nParts columns:")
print(parts.columns.tolist())


# Try to identify order and received date columns
order_candidates = [
    "order_date",
    "part_order_date",
    "parts_order_date",
    "ordered_date"
]

received_candidates = [
    "received_date",
    "part_received_date",
    "parts_received_date",
    "receipt_date"
]


order_col = next(
    (
        col for col in order_candidates
        if col in parts.columns
    ),
    None
)

received_col = next(
    (
        col for col in received_candidates
        if col in parts.columns
    ),
    None
)


# ============================================================
# 12. PARTS DELAY ANALYSIS
# ============================================================

if order_col and received_col:

    print(
        f"\nUsing parts dates: "
        f"{order_col} -> {received_col}"
    )

    parts[order_col] = pd.to_datetime(
        parts[order_col],
        errors="coerce"
    )

    parts[received_col] = pd.to_datetime(
        parts[received_col],
        errors="coerce"
    )

    parts["parts_delay_days"] = (
        parts[received_col] -
        parts[order_col]
    ).dt.days

    # More than 2 days = delayed part
    parts["part_delayed"] = (
        parts["parts_delay_days"] > 2
    )

    repair_parts = repairs[
        [
            "repair_id",
            "visit_id"
        ]
    ].merge(
        parts[
            [
                "repair_id",
                "parts_delay_days",
                "part_delayed"
            ]
        ],
        on="repair_id",
        how="left"
    )

    visit_parts = (
        repair_parts
        .groupby("visit_id")
        .agg(
            max_parts_delay_days=(
                "parts_delay_days",
                "max"
            ),
            parts_delayed=(
                "part_delayed",
                "max"
            )
        )
        .reset_index()
    )

    df = df.merge(
        visit_parts,
        on="visit_id",
        how="left"
    )

    df["parts_delayed"] = (
        df["parts_delayed"]
        .fillna(False)
    )


    # --------------------------------------------
    # Parts delay comparison
    # --------------------------------------------

    parts_summary = (
        df.groupby("parts_delayed")["delay_hours"]
        .agg(
            visits="count",
            average_delay="mean",
            median_delay="median"
        )
        .reset_index()
    )

    parts_summary.to_csv(
        f"{REPORT_PATH}/parts_delay_comparison.csv",
        index=False
    )

    print("\nParts Delay Comparison")
    print(parts_summary.to_string(index=False))


    # --------------------------------------------
    # Statistical test
    # --------------------------------------------

    # Make sure parts_delayed is a proper Boolean
df["parts_delayed"] = df["parts_delayed"].astype(bool)

# Make sure parts_delayed is a proper Boolean
df["parts_delayed"] = df["parts_delayed"].astype(bool)

df["parts_delayed"] = df["parts_delayed"].astype(bool)

no_parts_delay = df.loc[
    df["parts_delayed"] == False,
    "delay_hours"
].dropna()

with_parts_delay = df.loc[
    df["parts_delayed"] == True,
    "delay_hours"
].dropna()

with_parts_delay = df.loc[
        df["parts_delayed"],
        "delay_hours"
    ].dropna()


if (
        len(no_parts_delay) > 0
        and len(with_parts_delay) > 0
    ):

        u_stat, p_value = mannwhitneyu(
            no_parts_delay,
            with_parts_delay,
            alternative="two-sided"
        )

        test_3 = pd.DataFrame({
            "test": ["Mann-Whitney U"],
            "question": [
                "Is service delay different when parts are delayed?"
            ],
            "no_parts_delay_median": [
                no_parts_delay.median()
            ],
            "parts_delay_median": [
                with_parts_delay.median()
            ],
            "u_statistic": [
                u_stat
            ],
            "p_value": [
                p_value
            ],
            "significant_at_0_05": [
                p_value < 0.05
            ]
        })

        test_3.to_csv(
            f"{REPORT_PATH}/test_03_parts_delay.csv",
            index=False
        )

        print("\n----------------------------------------")
        print("TEST 3 — PARTS DELAY VS SERVICE DELAY")
        print("----------------------------------------")

        print(
            test_3.to_string(index=False)
        )


        # --------------------------------------------
        # Parts delay chart
        # --------------------------------------------

        plt.figure(figsize=(7, 5))

        plt.boxplot(
            [
                no_parts_delay,
                with_parts_delay
            ],
            tick_labels=[
                "No Parts Delay",
                "Parts Delayed"
            ]
        )

        plt.title(
            "Service Delay by Parts Delay Status"
        )

        plt.xlabel("Parts Status")
        plt.ylabel("Service Delay Hours")

        plt.tight_layout()

        plt.savefig(
            f"{CHART_PATH}/03_parts_delay_impact.png",
            dpi=150
        )

        plt.close()

else:

    print(
        "\nWARNING: Could not identify parts "
        "order/received date columns."
    )

    print(
        "Parts analysis skipped."
    )


# ============================================================
# 13. DEALER VARIABILITY
# ============================================================

print("\nCalculating dealer variability...")

dealer_summary = (
    df.groupby("dealer_id")
    .agg(
        visits=("visit_id", "count"),
        delay_rate=(
            "delayed",
            "mean"
        ),
        avg_delay_hours=(
            "delay_hours",
            "mean"
        ),
        avg_csat=(
            "csat_score",
            "mean"
        ),
        complaint_rate=(
            "complaint_flag",
            "mean"
        )
    )
    .reset_index()
)

dealer_summary.to_csv(
    f"{REPORT_PATH}/dealer_variability.csv",
    index=False
)


# ============================================================
# 14. SERVICE TYPE ANALYSIS
# ============================================================

print("Calculating service type performance...")

service_summary = (
    df.groupby("service_type")
    .agg(
        visits=("visit_id", "count"),
        delay_rate=(
            "delayed",
            "mean"
        ),
        avg_delay_hours=(
            "delay_hours",
            "mean"
        ),
        avg_csat=(
            "csat_score",
            "mean"
        ),
        complaint_rate=(
            "complaint_flag",
            "mean"
        )
    )
    .reset_index()
)

service_summary.to_csv(
    f"{REPORT_PATH}/service_type_summary.csv",
    index=False
)


# ============================================================
# 15. SAVE FINAL ANALYSIS DATASET
# ============================================================

df.to_csv(
    f"{REPORT_PATH}/analysis_dataset.csv",
    index=False
)


# ============================================================
# 16. FINAL OUTPUT
# ============================================================

print("\n========================================")
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("========================================")

print(
    f"\nReports saved to: {REPORT_PATH}"
)

print(
    f"Charts saved to: {CHART_PATH}"
)

print("\nKey files:")

print("1. overall_summary.csv")
print("2. test_01_csat_delay.csv")
print("3. test_02_delay_complaint.csv")
print("4. test_03_parts_delay.csv")
print("5. dealer_variability.csv")
print("6. service_type_summary.csv")
print("7. analysis_dataset.csv")