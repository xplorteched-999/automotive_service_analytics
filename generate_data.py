import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


# -----------------------------
# Configuration
# -----------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)


# Number of records
N_CUSTOMERS = 40_000
N_VEHICLES = 50_000
N_DEALERS = 200
N_APPOINTMENTS = 120_000
N_VISITS = 100_000
N_REPAIRS = 200_000
N_PARTS = 250_000
N_SURVEYS = 50_000


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DATA_DIR, exist_ok=True)


print("Configuration loaded successfully.")
print(f"Output folder: {RAW_DATA_DIR}")
# ============================================================
# 2.4 — GENERATE CUSTOMERS
# ============================================================

print("\n2.4 Generating customers...")

regions = ["North", "South", "East", "West", "Central"]
customer_types = ["Individual", "Fleet", "Corporate"]

customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUSTOMERS + 1),
    "customer_since": [
        fake.date_between(start_date="-10y", end_date="-30d")
        for _ in range(N_CUSTOMERS)
    ],
    "region": np.random.choice(
        regions,
        N_CUSTOMERS,
        p=[0.20, 0.25, 0.15, 0.25, 0.15]
    ),
    "customer_type": np.random.choice(
        customer_types,
        N_CUSTOMERS,
        p=[0.80, 0.10, 0.10]
    )
})

print(f"Customers: {len(customers):,}")


# ============================================================
# 2.5 — GENERATE VEHICLES
# ============================================================

print("\n2.5 Generating vehicles...")

vehicle_models = [
    "Compact Sedan",
    "Premium Sedan",
    "Compact SUV",
    "Mid SUV",
    "Large SUV",
    "Pickup",
    "Hatchback",
    "Electric SUV"
]

powertrains = [
    "Petrol",
    "Diesel",
    "Hybrid",
    "EV"
]

vehicle_customer_ids = np.random.choice(
    customers["customer_id"],
    N_VEHICLES
)

model_years = np.random.randint(2015, 2027, N_VEHICLES)

vehicles = pd.DataFrame({
    "vehicle_id": np.arange(1, N_VEHICLES + 1),
    "customer_id": vehicle_customer_ids,
    "vehicle_model": np.random.choice(vehicle_models, N_VEHICLES),
    "model_year": model_years,
    "powertrain": np.random.choice(
        powertrains,
        N_VEHICLES,
        p=[0.45, 0.25, 0.20, 0.10]
    )
})

vehicles["vehicle_age"] = 2026 - vehicles["model_year"]

vehicles["warranty_status"] = np.where(
    vehicles["vehicle_age"] <= 3,
    "In Warranty",
    "Out of Warranty"
)

print(f"Vehicles: {len(vehicles):,}")


# ============================================================
# 2.6 — GENERATE DEALERS
# ============================================================

print("\n2.6 Generating dealers...")

dealer_regions = np.random.choice(
    regions,
    N_DEALERS,
    p=[0.20, 0.25, 0.15, 0.25, 0.15]
)

dealers = pd.DataFrame({
    "dealer_id": np.arange(1, N_DEALERS + 1),
    "region": dealer_regions,
    "dealer_type": np.random.choice(
        ["Urban", "Suburban", "Rural"],
        N_DEALERS,
        p=[0.45, 0.40, 0.15]
    ),
    "service_bay_count": np.random.randint(8, 31, N_DEALERS),
    "technician_count": np.random.randint(10, 41, N_DEALERS)
})

print(f"Dealers: {len(dealers):,}")


# ============================================================
# 2.7 — GENERATE APPOINTMENTS
# ============================================================

print("\n2.7 Generating appointments...")

appointment_vehicle_ids = np.random.choice(
    vehicles["vehicle_id"],
    N_APPOINTMENTS
)

vehicle_lookup = vehicles.set_index("vehicle_id")

appointment_customer_ids = (
    vehicle_lookup.loc[
        appointment_vehicle_ids,
        "customer_id"
    ].values
)

appointment_dealer_ids = np.random.choice(
    dealers["dealer_id"],
    N_APPOINTMENTS
)

booking_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2025-01-01", "2026-06-30"),
        N_APPOINTMENTS
    )
)

lead_times = np.random.randint(1, 31, N_APPOINTMENTS)

appointment_dates = (
    booking_dates +
    pd.to_timedelta(lead_times, unit="D")
)

service_types = [
    "Scheduled Maintenance",
    "General Repair",
    "Brake Service",
    "Electrical",
    "AC Service",
    "Tire Service",
    "Warranty Repair",
    "Recall"
]

appointments = pd.DataFrame({
    "appointment_id": np.arange(1, N_APPOINTMENTS + 1),
    "customer_id": appointment_customer_ids,
    "vehicle_id": appointment_vehicle_ids,
    "dealer_id": appointment_dealer_ids,
    "booking_date": booking_dates,
    "appointment_date": appointment_dates,
    "service_type": np.random.choice(
        service_types,
        N_APPOINTMENTS,
        p=[0.25, 0.20, 0.12, 0.10, 0.08, 0.08, 0.10, 0.07]
    ),
    "scheduled_duration_hours": np.round(
        np.random.uniform(0.5, 5.0, N_APPOINTMENTS),
        2
    ),
    "appointment_status": np.random.choice(
        ["Completed", "Cancelled", "No Show"],
        N_APPOINTMENTS,
        p=[0.84, 0.10, 0.06]
    )
})

appointments["promised_completion"] = (
    appointments["appointment_date"] +
    pd.to_timedelta(
        appointments["scheduled_duration_hours"],
        unit="h"
    )
)

print(f"Appointments: {len(appointments):,}")


# ============================================================
# 2.8 — GENERATE SERVICE VISITS
# ============================================================

print("\n2.8 Generating service visits...")

completed_appointments = appointments[
    appointments["appointment_status"] == "Completed"
].copy()

completed_appointments = completed_appointments.sample(
    n=min(N_VISITS, len(completed_appointments)),
    random_state=SEED
).reset_index(drop=True)

n_actual_visits = len(completed_appointments)

check_in_times = (
    completed_appointments["appointment_date"] +
    pd.to_timedelta(
        np.random.uniform(-0.5, 1.5, n_actual_visits),
        unit="h"
    )
)

service_start_times = (
    check_in_times +
    pd.to_timedelta(
        np.random.uniform(0.1, 1.5, n_actual_visits),
        unit="h"
    )
)

# Base actual service duration
actual_duration = np.random.gamma(
    shape=2.5,
    scale=1.0,
    size=n_actual_visits
)

actual_duration = np.clip(actual_duration, 0.5, 12)

completion_times = (
    service_start_times +
    pd.to_timedelta(actual_duration, unit="h")
)

service_visits = pd.DataFrame({
    "visit_id": np.arange(1, n_actual_visits + 1),
    "appointment_id": completed_appointments["appointment_id"].values,
    "check_in_time": check_in_times,
    "service_start_time": service_start_times,
    "completion_time": completion_times,
    "visit_status": "Completed"
})

service_visits["actual_duration_hours"] = (
    service_visits["completion_time"] -
    service_visits["service_start_time"]
).dt.total_seconds() / 3600

service_visits["promised_completion"] = (
    completed_appointments["promised_completion"].values
)

service_visits["delay_hours"] = (
    service_visits["completion_time"] -
    service_visits["promised_completion"]
).dt.total_seconds() / 3600

service_visits["delay_flag"] = (
    service_visits["delay_hours"] > 0
).astype(int)

service_visits["first_time_fix_flag"] = np.random.choice(
    [0, 1],
    n_actual_visits,
    p=[0.18, 0.82]
)

print(f"Service visits: {len(service_visits):,}")


# ============================================================
# 2.9 — GENERATE REPAIRS
# ============================================================

print("\n2.9 Generating repairs...")

repair_visit_ids = np.random.choice(
    service_visits["visit_id"],
    N_REPAIRS
)

repair_categories = [
    "Engine",
    "Brakes",
    "Electrical",
    "AC",
    "Transmission",
    "Suspension",
    "Tires",
    "Routine Maintenance"
]

estimated_hours = np.round(
    np.random.uniform(0.5, 5.0, N_REPAIRS),
    2
)

actual_hours = np.round(
    estimated_hours *
    np.random.uniform(0.7, 2.0, N_REPAIRS),
    2
)

repairs = pd.DataFrame({
    "repair_id": np.arange(1, N_REPAIRS + 1),
    "visit_id": repair_visit_ids,
    "repair_category": np.random.choice(
        repair_categories,
        N_REPAIRS
    ),
    "estimated_hours": estimated_hours,
    "actual_hours": actual_hours,
    "repair_status": np.random.choice(
        ["Completed", "Additional Repair Required"],
        N_REPAIRS,
        p=[0.90, 0.10]
    )
})

print(f"Repairs: {len(repairs):,}")


# ============================================================
# 2.10 — GENERATE PARTS
# ============================================================

print("\n2.10 Generating parts transactions...")

part_repair_ids = np.random.choice(
    repairs["repair_id"],
    N_PARTS
)

part_order_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2025-01-01", "2026-06-30"),
        N_PARTS
    )
)

part_delay_days = np.random.choice(
    [0, 1, 2, 3, 5, 7, 10, 14],
    N_PARTS,
    p=[0.30, 0.20, 0.15, 0.12, 0.10, 0.06, 0.05, 0.02]
)

part_received_dates = (
    part_order_dates +
    pd.to_timedelta(part_delay_days, unit="D")
)

parts = pd.DataFrame({
    "part_transaction_id": np.arange(1, N_PARTS + 1),
    "repair_id": part_repair_ids,
    "part_id": np.random.randint(10000, 99999, N_PARTS),
    "part_order_date": part_order_dates,
    "part_received_date": part_received_dates
})

parts["parts_delay_flag"] = (
    part_delay_days > 2
).astype(int)

print(f"Parts transactions: {len(parts):,}")


# ============================================================
# 2.11 — GENERATE CUSTOMER SURVEYS
# ============================================================

print("\n2.11 Generating customer surveys...")

survey_visit_ids = np.random.choice(
    service_visits["visit_id"],
    N_SURVEYS,
    replace=False
)

survey_base = service_visits.set_index("visit_id").loc[
    survey_visit_ids
]

# Satisfaction is intentionally influenced by delay
survey_csat = []

for visit_id in survey_visit_ids:

    delay = survey_base.loc[
        visit_id,
        "delay_flag"
    ]

    if delay == 1:
        score = np.random.choice(
            [1, 2, 3, 4, 5],
            p=[0.20, 0.30, 0.30, 0.15, 0.05]
        )
    else:
        score = np.random.choice(
            [1, 2, 3, 4, 5],
            p=[0.03, 0.07, 0.15, 0.35, 0.40]
        )

    survey_csat.append(score)

customer_surveys = pd.DataFrame({
    "survey_id": np.arange(1, N_SURVEYS + 1),
    "visit_id": survey_visit_ids,
    "survey_date": pd.Timestamp("2026-06-30"),
    "csat_score": survey_csat
})

customer_surveys["complaint_flag"] = (
    customer_surveys["csat_score"] <= 2
).astype(int)

print(f"Customer surveys: {len(customer_surveys):,}")


# ============================================================
# 2.12 — CREATE REALISTIC BUSINESS RELATIONSHIPS
# ============================================================

print("\n2.12 Applying business relationships...")

# Higher dealer workload tends to increase delays
dealer_workload = dealers[
    ["dealer_id", "service_bay_count", "technician_count"]
].copy()

service_visits = service_visits.merge(
    completed_appointments[
        ["appointment_id", "dealer_id", "service_type"]
    ],
    on="appointment_id",
    how="left"
)

service_visits = service_visits.merge(
    dealer_workload,
    on="dealer_id",
    how="left"
)

# Dealers with fewer technicians receive a modest delay adjustment
workload_factor = (
    service_visits["service_bay_count"] /
    service_visits["technician_count"]
)

extra_delay = np.where(
    workload_factor < 0.45,
    np.random.uniform(0, 0.5, len(service_visits)),
    np.random.uniform(0, 1.5, len(service_visits))
)

service_visits["completion_time"] = (
    service_visits["completion_time"] +
    pd.to_timedelta(extra_delay, unit="h")
)

service_visits["actual_duration_hours"] = (
    service_visits["completion_time"] -
    service_visits["service_start_time"]
).dt.total_seconds() / 3600

service_visits["delay_hours"] = (
    service_visits["completion_time"] -
    service_visits["promised_completion"]
).dt.total_seconds() / 3600

service_visits["delay_flag"] = (
    service_visits["delay_hours"] > 0
).astype(int)

print("Business relationships applied.")


# ============================================================
# 2.13 — ADD INTENTIONAL DATA QUALITY ISSUES
# ============================================================

print("\n2.13 Adding intentional data-quality issues...")

# Missing customer IDs
missing_customer_rows = customers.sample(
    100,
    random_state=SEED
).index

customers.loc[
    missing_customer_rows,
    "customer_id"
] = np.nan


# Duplicate customer records
duplicate_customers = customers.iloc[
    100:110
].copy()

customers = pd.concat(
    [customers, duplicate_customers],
    ignore_index=True
)


# Invalid dealer IDs in appointments
invalid_dealer_rows = appointments.sample(
    100,
    random_state=SEED + 1
).index

appointments.loc[
    invalid_dealer_rows,
    "dealer_id"
] = 9999


# Invalid appointment dates
bad_date_rows = appointments.sample(
    100,
    random_state=SEED + 2
).index

appointments.loc[
    bad_date_rows,
    "appointment_date"
] = (
    appointments.loc[
        bad_date_rows,
        "booking_date"
    ] - pd.Timedelta(days=5)
)


# Missing CSAT
missing_csat_rows = customer_surveys.sample(
    250,
    random_state=SEED + 3
).index

customer_surveys.loc[
    missing_csat_rows,
    "csat_score"
] = np.nan


# Negative repair duration
negative_repair_rows = repairs.sample(
    100,
    random_state=SEED + 4
).index

repairs.loc[
    negative_repair_rows,
    "actual_hours"
] = -2


# Extreme repair-duration outliers
outlier_repair_rows = repairs.sample(
    50,
    random_state=SEED + 5
).index

repairs.loc[
    outlier_repair_rows,
    "actual_hours"
] = 100


# Missing parts received dates
missing_parts_rows = parts.sample(
    300,
    random_state=SEED + 6
).index

parts.loc[
    missing_parts_rows,
    "part_received_date"
] = pd.NaT


# Inconsistent category labels
category_rows = repairs.sample(
    300,
    random_state=SEED + 7
).index

repairs.loc[
    category_rows[:100],
    "repair_category"
] = "general repair"

repairs.loc[
    category_rows[100:200],
    "repair_category"
] = "GENERAL REPAIR"

repairs.loc[
    category_rows[200:],
    "repair_category"
] = "GEN_REPAIR"


print("Intentional data-quality issues added.")


# ============================================================
# 2.14 — SAVE ALL TABLES
# ============================================================

print("\n2.14 Saving CSV files...")

customers.to_csv(
    os.path.join(RAW_DATA_DIR, "customers.csv"),
    index=False
)

vehicles.to_csv(
    os.path.join(RAW_DATA_DIR, "vehicles.csv"),
    index=False
)

dealers.to_csv(
    os.path.join(RAW_DATA_DIR, "dealers.csv"),
    index=False
)

appointments.to_csv(
    os.path.join(RAW_DATA_DIR, "appointments.csv"),
    index=False
)

service_visits.to_csv(
    os.path.join(RAW_DATA_DIR, "service_visits.csv"),
    index=False
)

repairs.to_csv(
    os.path.join(RAW_DATA_DIR, "repairs.csv"),
    index=False
)

parts.to_csv(
    os.path.join(RAW_DATA_DIR, "parts.csv"),
    index=False
)

customer_surveys.to_csv(
    os.path.join(RAW_DATA_DIR, "customer_surveys.csv"),
    index=False
)

print("All CSV files saved.")


# ============================================================
# 2.15 — BASIC VALIDATION
# ============================================================

print("\n2.15 Running validation...")

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

print("\nRow counts:")
print("-" * 45)

for table_name, df in tables.items():
    print(f"{table_name:<25} {len(df):>10,}")


print("\nColumn counts:")
print("-" * 45)

for table_name, df in tables.items():
    print(f"{table_name:<25} {len(df.columns):>10}")


# Check important relationships

valid_vehicle_customers = vehicles["customer_id"].isin(
    customers["customer_id"].dropna()
).mean()

valid_appointment_dealers = appointments["dealer_id"].isin(
    dealers["dealer_id"]
).mean()

valid_repair_visits = repairs["visit_id"].isin(
    service_visits["visit_id"]
).mean()

valid_part_repairs = parts["repair_id"].isin(
    repairs["repair_id"]
).mean()

valid_survey_visits = customer_surveys["visit_id"].isin(
    service_visits["visit_id"]
).mean()

print("\nRelationship checks:")
print("-" * 45)

print(
    f"Vehicles → Customers: "
    f"{valid_vehicle_customers:.2%}"
)

print(
    f"Appointments → Dealers: "
    f"{valid_appointment_dealers:.2%}"
)

print(
    f"Repairs → Service Visits: "
    f"{valid_repair_visits:.2%}"
)

print(
    f"Parts → Repairs: "
    f"{valid_part_repairs:.2%}"
)

print(
    f"Surveys → Service Visits: "
    f"{valid_survey_visits:.2%}"
)


# ============================================================
# 2.16 — FINAL PHASE 2 CHECK
# ============================================================

print("\n2.16 FINAL PHASE 2 CHECK")
print("=" * 60)

expected_files = [
    "customers.csv",
    "vehicles.csv",
    "dealers.csv",
    "appointments.csv",
    "service_visits.csv",
    "repairs.csv",
    "parts.csv",
    "customer_surveys.csv"
]

all_files_exist = True

for filename in expected_files:

    filepath = os.path.join(
        RAW_DATA_DIR,
        filename
    )

    exists = os.path.exists(filepath)

    print(
        f"{filename:<30} "
        f"{'OK' if exists else 'MISSING'}"
    )

    if not exists:
        all_files_exist = False


print("\n" + "=" * 60)

if all_files_exist:
    print("PHASE 2 COMPLETE — ALL DATA FILES GENERATED SUCCESSFULLY.")
else:
    print("PHASE 2 INCOMPLETE — CHECK MISSING FILES.")

print("=" * 60)
