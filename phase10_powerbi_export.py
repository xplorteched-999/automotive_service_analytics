import os
import pandas as pd

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase10_powerbi"

os.makedirs(REPORT_PATH, exist_ok=True)

print("======================================")
print("PHASE 10 — POWER BI DATA EXPORT")
print("======================================")

# ============================================================
# 1. LOAD DATA
# ============================================================

customers = pd.read_csv(
    f"{DATA_PATH}/customers.csv"
)

vehicles = pd.read_csv(
    f"{DATA_PATH}/vehicles.csv"
)

appointments = pd.read_csv(
    f"{DATA_PATH}/appointments.csv"
)

visits = pd.read_csv(
    f"{DATA_PATH}/service_visits.csv"
)

dealers = pd.read_csv(
    f"{DATA_PATH}/dealers.csv"
)

surveys = pd.read_csv(
    f"{DATA_PATH}/customer_surveys.csv"
)

segments = pd.read_csv(
    "reports/phase9/customer_segments_refined.csv"
)

segment_profiles = pd.read_csv(
    "reports/phase9/segment_profiles_refined.csv"
)

experiment = pd.read_csv(
    "reports/phase8/experiment_dataset.csv"
)

# ============================================================
# 2. FIX SEGMENT COLUMN
# ============================================================

# Phase 9 refinement creates:
# cluster_id_refined
# segment_name

if "segment_name" not in segments.columns:

    print("\nsegment_name not found in customer segment file.")

    if "segment_name_y" in segments.columns:
        segments["segment_name"] = segments["segment_name_y"]

    elif "segment_name_x" in segments.columns:
        segments["segment_name"] = segments["segment_name_x"]

    else:
        raise ValueError(
            "No segment_name column found in "
            "customer_segments_refined.csv"
        )

# ============================================================
# 3. CREATE SERVICE FACT TABLE
# ============================================================

service_fact = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "vehicle_id",
            "booking_date",
            "appointment_date",
            "service_type",
            "scheduled_duration_hours",
            "appointment_status"
        ]
    ],
    on="appointment_id",
    how="left"
)

service_fact = service_fact.merge(
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

service_fact = service_fact.merge(
    dealers,
    on="dealer_id",
    how="left"
)

# ============================================================
# 4. DATE FIELDS
# ============================================================

service_fact["check_in_time"] = pd.to_datetime(
    service_fact["check_in_time"],
    errors="coerce"
)

service_fact["appointment_date"] = pd.to_datetime(
    service_fact["appointment_date"],
    errors="coerce"
)

service_fact["booking_date"] = pd.to_datetime(
    service_fact["booking_date"],
    errors="coerce"
)

service_fact["service_date"] = (
    service_fact["check_in_time"].dt.date
)

service_fact["service_year"] = (
    service_fact["check_in_time"].dt.year
)

service_fact["service_month"] = (
    service_fact["check_in_time"].dt.month
)

service_fact["service_month_name"] = (
    service_fact["check_in_time"].dt.strftime("%b")
)

service_fact["service_week"] = (
    service_fact["check_in_time"].dt.isocalendar().week
)

service_fact["day_of_week"] = (
    service_fact["check_in_time"].dt.day_name()
)

# ============================================================
# 5. BUSINESS FLAGS
# ============================================================

service_fact["delayed_visit"] = (
    service_fact["delay_hours"] > 0
).astype(int)

service_fact["severe_delay"] = (
    service_fact["delay_hours"] >= 4
).astype(int)

service_fact["first_time_fix"] = (
    service_fact["first_time_fix_flag"] == 1
).astype(int)

# ============================================================
# 6. CUSTOMER DIMENSION
# ============================================================

segment_columns = [
    "customer_id",
    "segment_name",
    "visit_count",
    "delay_rate",
    "avg_csat",
    "complaint_rate",
    "recency_days"
]

customer_dimension = customers.merge(
    segments[segment_columns],
    on="customer_id",
    how="left"
)

# ============================================================
# 7. OTHER DIMENSIONS
# ============================================================

vehicle_dimension = vehicles.copy()

dealer_dimension = dealers.copy()

# ============================================================
# 8. SEGMENT SUMMARY
# ============================================================

segment_summary = segment_profiles.copy()

# ============================================================
# 9. EXPERIMENT SUMMARY
# ============================================================

experiment_summary = (
    experiment
    .groupby("experiment_group")
    .agg(
        customers=("experiment_group", "count"),
        delay_rate=("final_delay_flag", "mean"),
        avg_csat=("csat_score", "mean"),
        complaint_rate=("complaint_flag", "mean")
    )
    .reset_index()
)

# ============================================================
# 10. SAVE POWER BI TABLES
# ============================================================

tables = {
    "fact_service_visits.csv": service_fact,
    "dim_customers.csv": customer_dimension,
    "dim_vehicles.csv": vehicle_dimension,
    "dim_dealers.csv": dealer_dimension,
    "segment_summary.csv": segment_summary,
    "experiment_summary.csv": experiment_summary
}

for filename, dataframe in tables.items():

    path = f"{REPORT_PATH}/{filename}"

    dataframe.to_csv(
        path,
        index=False
    )

    print(
        f"Exported {filename}: "
        f"{dataframe.shape}"
    )

print("\n======================================")
print("POWER BI EXPORT COMPLETED")
print("======================================")

print(
    f"\nFiles saved to: {REPORT_PATH}"
)