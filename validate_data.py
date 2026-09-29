import os
import pandas as pd
import numpy as np


# ============================================================
# PHASE 3 — DATA QUALITY & CLEANING
# ============================================================

SEED = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# 3.2 — LOAD ALL RAW TABLES
# ============================================================

print("=" * 70)
print("PHASE 3 — DATA QUALITY & CLEANING")
print("=" * 70)

print("\n3.2 Loading raw data...")

customers = pd.read_csv(
    os.path.join(RAW_DIR, "customers.csv")
)

vehicles = pd.read_csv(
    os.path.join(RAW_DIR, "vehicles.csv")
)

dealers = pd.read_csv(
    os.path.join(RAW_DIR, "dealers.csv")
)

appointments = pd.read_csv(
    os.path.join(RAW_DIR, "appointments.csv")
)

service_visits = pd.read_csv(
    os.path.join(RAW_DIR, "service_visits.csv")
)

repairs = pd.read_csv(
    os.path.join(RAW_DIR, "repairs.csv")
)

parts = pd.read_csv(
    os.path.join(RAW_DIR, "parts.csv")
)

customer_surveys = pd.read_csv(
    os.path.join(RAW_DIR, "customer_surveys.csv")
)


tables = {
    "customers": customers,
    "vehicles": vehicles,
    "dealers": dealers,
    "appointments": appointments,
    "service_visits": service_visits,
    "repairs": repairs,
    "parts": parts,
    "customer_surveys": customer_surveys
}

print("All 8 tables loaded successfully.")


# ============================================================
# 3.3 — DATA PROFILING
# ============================================================

print("\n3.3 Profiling tables...")

profile_results = []

for table_name, df in tables.items():

    for column in df.columns:

        profile_results.append({
            "table": table_name,
            "column": column,
            "data_type": str(df[column].dtype),
            "row_count": len(df),
            "missing_count": int(df[column].isna().sum()),
            "missing_pct": round(
                df[column].isna().mean() * 100,
                2
            ),
            "unique_count": int(
                df[column].nunique(dropna=True)
            )
        })


profile_df = pd.DataFrame(profile_results)

profile_path = os.path.join(
    REPORT_DIR,
    "data_profile.csv"
)

profile_df.to_csv(
    profile_path,
    index=False
)

print(f"Profile saved: {profile_path}")


# ============================================================
# 3.4 — DETECT MISSING VALUES
# ============================================================

print("\n3.4 Detecting missing values...")

quality_issues = []


def add_issue(
    table,
    column,
    issue_type,
    affected_rows,
    severity,
    action
):
    quality_issues.append({
        "table": table,
        "column": column,
        "issue_type": issue_type,
        "affected_rows": affected_rows,
        "severity": severity,
        "recommended_action": action
    })


for table_name, df in tables.items():

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count > 0:

            severity = (
                "High"
                if missing_count / len(df) > 0.05
                else "Medium"
            )

            add_issue(
                table_name,
                column,
                "Missing Values",
                missing_count,
                severity,
                "Investigate and handle based on business meaning"
            )


# ============================================================
# 3.5 — DETECT DUPLICATES
# ============================================================

print("3.5 Detecting duplicates...")

primary_keys = {
    "customers": "customer_id",
    "vehicles": "vehicle_id",
    "dealers": "dealer_id",
    "appointments": "appointment_id",
    "service_visits": "visit_id",
    "repairs": "repair_id",
    "parts": "part_transaction_id",
    "customer_surveys": "survey_id"
}


for table_name, key in primary_keys.items():

    df = tables[table_name]

    duplicate_count = int(
        df[key].duplicated(
            keep=False
        ).sum()
    )

    if duplicate_count > 0:

        add_issue(
            table_name,
            key,
            "Duplicate Primary Key",
            duplicate_count,
            "High",
            "Remove duplicate records after investigation"
        )


# ============================================================
# 3.6 — DETECT INVALID DATES
# ============================================================

print("3.6 Checking date logic...")

appointments["booking_date"] = pd.to_datetime(
    appointments["booking_date"],
    errors="coerce"
)

appointments["appointment_date"] = pd.to_datetime(
    appointments["appointment_date"],
    errors="coerce"
)

service_visits["check_in_time"] = pd.to_datetime(
    service_visits["check_in_time"],
    errors="coerce"
)

service_visits["service_start_time"] = pd.to_datetime(
    service_visits["service_start_time"],
    errors="coerce"
)

service_visits["completion_time"] = pd.to_datetime(
    service_visits["completion_time"],
    errors="coerce"
)

service_visits["promised_completion"] = pd.to_datetime(
    service_visits["promised_completion"],
    errors="coerce"
)

parts["part_order_date"] = pd.to_datetime(
    parts["part_order_date"],
    errors="coerce"
)

parts["part_received_date"] = pd.to_datetime(
    parts["part_received_date"],
    errors="coerce"
)


invalid_appointment_dates = (
    appointments["appointment_date"] <
    appointments["booking_date"]
)

count_invalid_dates = int(
    invalid_appointment_dates.sum()
)

if count_invalid_dates > 0:

    add_issue(
        "appointments",
        "appointment_date",
        "Appointment Before Booking",
        count_invalid_dates,
        "High",
        "Remove invalid appointment records"
    )


# Service start should not be after completion
invalid_service_times = (
    service_visits["service_start_time"] >
    service_visits["completion_time"]
)

count_invalid_service_times = int(
    invalid_service_times.sum()
)

if count_invalid_service_times > 0:

    add_issue(
        "service_visits",
        "service_start_time",
        "Start After Completion",
        count_invalid_service_times,
        "High",
        "Remove or investigate invalid service records"
    )


# Parts received before ordered
invalid_parts_dates = (
    parts["part_received_date"].notna()
    &
    (
        parts["part_received_date"] <
        parts["part_order_date"]
    )
)

count_invalid_parts_dates = int(
    invalid_parts_dates.sum()
)

if count_invalid_parts_dates > 0:

    add_issue(
        "parts",
        "part_received_date",
        "Parts Received Before Ordered",
        count_invalid_parts_dates,
        "High",
        "Remove or correct invalid dates"
    )


# ============================================================
# 3.7 — FOREIGN KEY VALIDATION
# ============================================================

print("3.7 Checking relationships...")


def foreign_key_check(
    child_df,
    child_column,
    parent_df,
    parent_column,
    child_name
):

    valid_values = set(
        parent_df[parent_column]
        .dropna()
        .unique()
    )

    invalid_mask = (
        ~child_df[child_column].isin(valid_values)
        &
        child_df[child_column].notna()
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count > 0:

        add_issue(
            child_name,
            child_column,
            "Invalid Foreign Key",
            invalid_count,
            "High",
            "Remove or correct orphan records"
        )

    return invalid_count


foreign_key_check(
    vehicles,
    "customer_id",
    customers,
    "customer_id",
    "vehicles"
)

foreign_key_check(
    appointments,
    "customer_id",
    customers,
    "customer_id",
    "appointments"
)

foreign_key_check(
    appointments,
    "vehicle_id",
    vehicles,
    "vehicle_id",
    "appointments"
)

foreign_key_check(
    appointments,
    "dealer_id",
    dealers,
    "dealer_id",
    "appointments"
)

foreign_key_check(
    service_visits,
    "appointment_id",
    appointments,
    "appointment_id",
    "service_visits"
)

foreign_key_check(
    repairs,
    "visit_id",
    service_visits,
    "visit_id",
    "repairs"
)

foreign_key_check(
    parts,
    "repair_id",
    repairs,
    "repair_id",
    "parts"
)

foreign_key_check(
    customer_surveys,
    "visit_id",
    service_visits,
    "visit_id",
    "customer_surveys"
)


# ============================================================
# 3.8 — NUMERIC VALIDATION
# ============================================================

print("3.8 Checking numeric values and outliers...")


# Negative repair hours
negative_repair_hours = (
    repairs["actual_hours"] < 0
)

count_negative_repair = int(
    negative_repair_hours.sum()
)

if count_negative_repair > 0:

    add_issue(
        "repairs",
        "actual_hours",
        "Negative Duration",
        count_negative_repair,
        "High",
        "Set invalid values to missing and investigate"
    )


# Extreme repair duration
extreme_repair_hours = (
    repairs["actual_hours"] > 24
)

count_extreme_repair = int(
    extreme_repair_hours.sum()
)

if count_extreme_repair > 0:

    add_issue(
        "repairs",
        "actual_hours",
        "Extreme Outlier",
        count_extreme_repair,
        "Medium",
        "Treat extreme values as invalid/outliers"
    )


# Negative scheduled duration
negative_scheduled_duration = (
    appointments["scheduled_duration_hours"] <= 0
)

count_negative_scheduled = int(
    negative_scheduled_duration.sum()
)

if count_negative_scheduled > 0:

    add_issue(
        "appointments",
        "scheduled_duration_hours",
        "Invalid Duration",
        count_negative_scheduled,
        "High",
        "Remove invalid appointments"
    )


# Invalid CSAT
invalid_csat = (
    ~customer_surveys["csat_score"].isin(
        [1, 2, 3, 4, 5]
    )
    &
    customer_surveys["csat_score"].notna()
)

count_invalid_csat = int(
    invalid_csat.sum()
)

if count_invalid_csat > 0:

    add_issue(
        "customer_surveys",
        "csat_score",
        "Invalid CSAT",
        count_invalid_csat,
        "High",
        "Set invalid values to missing"
    )


# ============================================================
# 3.9 — CATEGORY STANDARDIZATION CHECK
# ============================================================

print("3.9 Checking categorical consistency...")

repair_categories_before = sorted(
    repairs["repair_category"]
    .dropna()
    .unique()
)

print("\nRepair categories before standardization:")
for category in repair_categories_before:
    print(f"  - {category}")


add_issue(
    "repairs",
    "repair_category",
    "Inconsistent Category Labels",
    int(
        repairs["repair_category"]
        .isin(
            ["general repair", "GENERAL REPAIR", "GEN_REPAIR"]
        )
        .sum()
    ),
    "Medium",
    "Standardize category labels"
)


# ============================================================
# 3.10 — CREATE DATA QUALITY REPORT
# ============================================================

print("\n3.10 Creating data-quality report...")

quality_report = pd.DataFrame(
    quality_issues
)

quality_report_path = os.path.join(
    REPORT_DIR,
    "data_quality_report.csv"
)

quality_report.to_csv(
    quality_report_path,
    index=False
)

print(
    f"Data-quality report saved: "
    f"{quality_report_path}"
)

print(
    f"Total quality issues detected: "
    f"{len(quality_report):,}"
)


# ============================================================
# ============================================================
# 3.11 — CLEAN CUSTOMERS
# ============================================================

print("\n3.11 Cleaning customers...")

before = len(customers)

# Remove customers without a valid primary key
customers = customers[
    customers["customer_id"].notna()
].copy()

# Convert ID to integer
customers["customer_id"] = (
    customers["customer_id"].astype(int)
)

# Remove duplicate customer IDs
customers = customers.drop_duplicates(
    subset=["customer_id"],
    keep="first"
)

after = len(customers)

print(
    f"Customers: {before:,} → {after:,}"
)


# ============================================================
# 3.12 — CLEAN VEHICLES
# ============================================================

print("\n3.12 Cleaning vehicles...")

before = len(vehicles)

# Remove vehicles with missing IDs
vehicles = vehicles[
    vehicles["vehicle_id"].notna()
].copy()

vehicles["vehicle_id"] = (
    vehicles["vehicle_id"].astype(int)
)

# Keep only vehicles belonging to valid customers
valid_customer_ids = set(
    customers["customer_id"]
)

vehicles = vehicles[
    vehicles["customer_id"].isin(valid_customer_ids)
].copy()

after = len(vehicles)

print(
    f"Vehicles: {before:,} → {after:,}"
)


# ============================================================
# 3.13 — CLEAN APPOINTMENTS
# ============================================================

print("\n3.13 Cleaning appointments...")

before = len(appointments)

# Remove appointments with missing IDs
appointments = appointments[
    appointments["appointment_id"].notna()
].copy()

appointments["appointment_id"] = (
    appointments["appointment_id"].astype(int)
)

# Remove invalid appointment dates
appointments = appointments[
    appointments["appointment_date"] >=
    appointments["booking_date"]
].copy()

# Valid customers
valid_customer_ids = set(
    customers["customer_id"]
)

appointments = appointments[
    appointments["customer_id"].isin(
        valid_customer_ids
    )
].copy()

# Valid vehicles
valid_vehicle_ids = set(
    vehicles["vehicle_id"]
)

appointments = appointments[
    appointments["vehicle_id"].isin(
        valid_vehicle_ids
    )
].copy()

# Valid dealers
valid_dealer_ids = set(
    dealers["dealer_id"]
)

appointments = appointments[
    appointments["dealer_id"].isin(
        valid_dealer_ids
    )
].copy()

# Remove invalid scheduled durations
appointments = appointments[
    appointments["scheduled_duration_hours"] > 0
].copy()

after = len(appointments)

print(
    f"Appointments: {before:,} → {after:,}"
)


# ============================================================
# 3.14 — CLEAN SERVICE VISITS
# ============================================================

print("\n3.14 Cleaning service visits...")

before = len(service_visits)

# Remove visits with missing IDs
service_visits = service_visits[
    service_visits["visit_id"].notna()
].copy()

service_visits["visit_id"] = (
    service_visits["visit_id"].astype(int)
)

# Keep only visits belonging to valid appointments
valid_appointment_ids = set(
    appointments["appointment_id"]
)

service_visits = service_visits[
    service_visits["appointment_id"].isin(
        valid_appointment_ids
    )
].copy()

# Remove invalid service time records
service_visits = service_visits[
    service_visits["service_start_time"] <=
    service_visits["completion_time"]
].copy()

# Recalculate actual duration
service_visits["actual_duration_hours"] = (
    service_visits["completion_time"] -
    service_visits["service_start_time"]
).dt.total_seconds() / 3600

# Recalculate delay
service_visits["delay_hours"] = (
    service_visits["completion_time"] -
    service_visits["promised_completion"]
).dt.total_seconds() / 3600

# Recalculate delay flag
service_visits["delay_flag"] = (
    service_visits["delay_hours"] > 0
).astype(int)

after = len(service_visits)

print(
    f"Service visits: {before:,} → {after:,}"
)


# ============================================================
# 3.15 — CLEAN REPAIRS
# ============================================================

print("\n3.15 Cleaning repairs...")

before = len(repairs)

# Remove repairs with missing IDs
repairs = repairs[
    repairs["repair_id"].notna()
].copy()

repairs["repair_id"] = (
    repairs["repair_id"].astype(int)
)

# Keep only repairs belonging to FINAL valid service visits
valid_visit_ids = set(
    service_visits["visit_id"]
)

repairs = repairs[
    repairs["visit_id"].isin(
        valid_visit_ids
    )
].copy()

# Negative repair durations → missing
repairs.loc[
    repairs["actual_hours"] < 0,
    "actual_hours"
] = np.nan

# Extreme repair durations → missing
repairs.loc[
    repairs["actual_hours"] > 24,
    "actual_hours"
] = np.nan

# Standardize repair categories
category_mapping = {
    "general repair": "General Repair",
    "GENERAL REPAIR": "General Repair",
    "GEN_REPAIR": "General Repair"
}

repairs["repair_category"] = (
    repairs["repair_category"]
    .replace(category_mapping)
)

after = len(repairs)

print(
    f"Repairs: {before:,} → {after:,}"
)


# ============================================================
# 3.16 — CLEAN PARTS
# ============================================================

print("\n3.16 Cleaning parts...")

before = len(parts)

# Remove parts with missing IDs
parts = parts[
    parts["part_transaction_id"].notna()
].copy()

parts["part_transaction_id"] = (
    parts["part_transaction_id"].astype(int)
)

# Keep only parts belonging to FINAL valid repairs
valid_repair_ids = set(
    repairs["repair_id"]
)

parts = parts[
    parts["repair_id"].isin(
        valid_repair_ids
    )
].copy()

# Recalculate invalid received dates
invalid_parts_dates_clean = (
    parts["part_received_date"].notna()
    &
    (
        parts["part_received_date"] <
        parts["part_order_date"]
    )
)

parts.loc[
    invalid_parts_dates_clean,
    "part_received_date"
] = pd.NaT

# Recalculate parts delay flag
parts["parts_delay_flag"] = np.where(
    parts["part_received_date"].isna(),
    np.nan,
    (
        (
            parts["part_received_date"] -
            parts["part_order_date"]
        ).dt.days > 2
    ).astype(int)
)

after = len(parts)

print(
    f"Parts: {before:,} → {after:,}"
)


# ============================================================
# 3.17 — CLEAN CUSTOMER SURVEYS
# ============================================================

print("\n3.17 Cleaning customer surveys...")

before = len(customer_surveys)

# Remove surveys with missing IDs
customer_surveys = customer_surveys[
    customer_surveys["survey_id"].notna()
].copy()

customer_surveys["survey_id"] = (
    customer_surveys["survey_id"].astype(int)
)

# Keep only surveys belonging to FINAL valid service visits
valid_visit_ids = set(
    service_visits["visit_id"]
)

customer_surveys = customer_surveys[
    customer_surveys["visit_id"].isin(
        valid_visit_ids
    )
].copy()

# Invalid CSAT → missing
invalid_csat_clean = (
    ~customer_surveys["csat_score"].isin(
        [1, 2, 3, 4, 5]
    )
    &
    customer_surveys["csat_score"].notna()
)

customer_surveys.loc[
    invalid_csat_clean,
    "csat_score"
] = np.nan

# Recalculate complaint flag
customer_surveys["complaint_flag"] = np.where(
    customer_surveys["csat_score"].isna(),
    np.nan,
    (
        customer_surveys["csat_score"] <= 2
    ).astype(int)
)

after = len(customer_surveys)

print(
    f"Customer surveys: {before:,} → {after:,}"
)


# ============================================================
# 3.18 — FINAL CASCADE CLEANUP
# ============================================================
#
# This is important.
#
# Service visits were cleaned after the initial repair cleanup.
# Therefore, repairs, parts and surveys are filtered one final
# time against the FINAL parent tables.
#
# This prevents orphan records caused by later parent deletions.
# ============================================================

print("\n3.18 Performing final cascade cleanup...")

# ------------------------------------------------------------
# Repairs → Service Visits
# ------------------------------------------------------------

valid_visit_ids = set(
    service_visits["visit_id"]
)

repairs = repairs[
    repairs["visit_id"].isin(
        valid_visit_ids
    )
].copy()


# ------------------------------------------------------------
# Parts → Repairs
# ------------------------------------------------------------

valid_repair_ids = set(
    repairs["repair_id"]
)

parts = parts[
    parts["repair_id"].isin(
        valid_repair_ids
    )
].copy()


# ------------------------------------------------------------
# Surveys → Service Visits
# ------------------------------------------------------------

customer_surveys = customer_surveys[
    customer_surveys["visit_id"].isin(
        valid_visit_ids
    )
].copy()


print(
    f"Final repairs rows: "
    f"{len(repairs):,}"
)

print(
    f"Final parts rows: "
    f"{len(parts):,}"
)

print(
    f"Final survey rows: "
    f"{len(customer_surveys):,}"
)


# ============================================================
# 3.19 — FINAL RELATIONSHIP VALIDATION
# ============================================================

print("\n3.19 Final relationship validation...")

clean_tables = {
    "customers": customers,
    "vehicles": vehicles,
    "dealers": dealers,
    "appointments": appointments,
    "service_visits": service_visits,
    "repairs": repairs,
    "parts": parts,
    "customer_surveys": customer_surveys
}


checks = [
    (
        "vehicles",
        "customer_id",
        "customers",
        "customer_id"
    ),
    (
        "appointments",
        "customer_id",
        "customers",
        "customer_id"
    ),
    (
        "appointments",
        "vehicle_id",
        "vehicles",
        "vehicle_id"
    ),
    (
        "appointments",
        "dealer_id",
        "dealers",
        "dealer_id"
    ),
    (
        "service_visits",
        "appointment_id",
        "appointments",
        "appointment_id"
    ),
    (
        "repairs",
        "visit_id",
        "service_visits",
        "visit_id"
    ),
    (
        "parts",
        "repair_id",
        "repairs",
        "repair_id"
    ),
    (
        "customer_surveys",
        "visit_id",
        "service_visits",
        "visit_id"
    )
]


relationship_results = []


for (
    child_table,
    child_column,
    parent_table,
    parent_column
) in checks:

    child_df = clean_tables[child_table]
    parent_df = clean_tables[parent_table]

    # Ignore null foreign keys
    child_values = (
        child_df[child_column]
        .dropna()
    )

    parent_values = set(
        parent_df[parent_column]
        .dropna()
    )

    valid_count = (
        child_values
        .isin(parent_values)
        .sum()
    )

    total_count = len(child_values)

    if total_count == 0:

        valid_pct = 100.0

    else:

        valid_pct = (
            valid_count /
            total_count *
            100
        )

    relationship_results.append({

        "child_table":
            child_table,

        "child_column":
            child_column,

        "parent_table":
            parent_table,

        "parent_column":
            parent_column,

        "valid_relationship_pct":
            round(
                valid_pct,
                2
            )
    })


relationship_df = pd.DataFrame(
    relationship_results
)


# ============================================================
# 3.20 — EXPLICIT ORPHAN CHECK
# ============================================================

print("\nExplicit orphan check...")

for (
    child_table,
    child_column,
    parent_table,
    parent_column
) in checks:

    child_df = clean_tables[child_table]
    parent_df = clean_tables[parent_table]

    child_values = (
        child_df[child_column]
        .dropna()
    )

    parent_values = set(
        parent_df[parent_column]
        .dropna()
    )

    orphan_count = (
        ~child_values.isin(parent_values)
    ).sum()

    print(
        f"{child_table}.{child_column} → "
        f"{parent_table}.{parent_column}: "
        f"{orphan_count:,} orphan rows"
    )


# ============================================================
# 3.21 — SAVE RELATIONSHIP REPORT
# ============================================================

relationship_path = os.path.join(
    REPORT_DIR,
    "post_cleaning_relationships.csv"
)

relationship_df.to_csv(
    relationship_path,
    index=False
)


# ============================================================
# 3.22 — DISPLAY RELATIONSHIP RESULTS
# ============================================================

print("\nPost-cleaning relationship results:")

print(
    relationship_df.to_string(
        index=False
    )
)


# ============================================================
# 3.23 — FINAL INTEGRITY CHECK
# ============================================================

failed_relationships = (
    relationship_df[
        relationship_df[
            "valid_relationship_pct"
        ] < 100
    ]
)


if len(failed_relationships) > 0:

    print("\nWARNING:")
    print(
        "Some relationships are still below 100%."
    )

    print(
        failed_relationships.to_string(
            index=False
        )
    )

    raise ValueError(
        "Phase 3 failed: orphan relationships remain."
    )

else:

    print(
        "\nALL RELATIONSHIPS VALID: 100%"
    )


# ============================================================
# 3.24 — SAVE CLEANED DATASETS
# ============================================================

print("\n3.24 Saving cleaned datasets...")

for table_name, df in clean_tables.items():

    output_path = os.path.join(
        PROCESSED_DIR,
        f"{table_name}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"{table_name:<25}"
        f"{len(df):>10,} rows"
    )


# ============================================================
# 3.25 — FINAL PHASE 3 SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

print("\nCleaned datasets:")
print(
    f"Location: {PROCESSED_DIR}"
)

print("\nReports generated:")

print(
    f"1. {profile_path}"
)

print(
    f"2. {quality_report_path}"
)

print(
    f"3. {relationship_path}"
)

print("\nFinal row counts:")

for table_name, df in clean_tables.items():

    print(
        f"{table_name:<25}"
        f"{len(df):>10,}"
    )

print("\n" + "=" * 70)
print(
    "DATA QUALITY PIPELINE FINISHED SUCCESSFULLY"
)
print("=" * 70)