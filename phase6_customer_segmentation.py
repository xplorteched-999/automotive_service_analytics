import os
import pandas as pd
import numpy as np

# ============================================================
# 1. SETUP
# ============================================================

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase6"

os.makedirs(REPORT_PATH, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

visits = pd.read_csv(
    f"{DATA_PATH}/service_visits.csv"
)

appointments = pd.read_csv(
    f"{DATA_PATH}/appointments.csv"
)

surveys = pd.read_csv(
    f"{DATA_PATH}/customer_surveys.csv"
)

print("Data loaded")
print("Visits:", visits.shape)
print("Appointments:", appointments.shape)
print("Surveys:", surveys.shape)


# ============================================================
# 3. BUILD CUSTOMER-LEVEL DATASET
# ============================================================

df = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "dealer_id",
            "service_type"
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

print("\nVisit-level dataset:", df.shape)


# ============================================================
# 4. CREATE CUSTOMER FEATURES
# ============================================================

customer = (
    df.groupby("customer_id")
    .agg(
        total_visits=("visit_id", "count"),

        avg_delay_hours=("delay_hours", "mean"),

        delay_rate=(
            "delay_hours",
            lambda x: (x > 0).mean()
        ),

        avg_csat=("csat_score", "mean"),

        complaint_rate=("complaint_flag", "mean"),

        total_complaints=("complaint_flag", "sum"),

        avg_service_duration=(
            "actual_duration_hours",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

customer["avg_csat"] = customer["avg_csat"].fillna(
    customer["avg_csat"].median()
)

customer["complaint_rate"] = customer[
    "complaint_rate"
].fillna(0)

customer["avg_delay_hours"] = customer[
    "avg_delay_hours"
].fillna(0)

customer["avg_service_duration"] = customer[
    "avg_service_duration"
].fillna(
    customer["avg_service_duration"].median()
)


# ============================================================
# # ============================================================
# 6. CREATE CUSTOMER SEGMENTS
# ============================================================

# Calculate population thresholds
high_delay = customer["delay_rate"].quantile(0.75)
high_complaint = customer["complaint_rate"].quantile(0.75)
low_csat = customer["avg_csat"].quantile(0.25)
high_visits = customer["total_visits"].quantile(0.75)
low_delay = customer["delay_rate"].quantile(0.25)
high_csat = customer["avg_csat"].quantile(0.75)

customer["segment"] = np.select(

    [
        # At-Risk:
        # poor customer experience + operational problems
        (
            (customer["avg_csat"] <= low_csat) &
            (
                (customer["delay_rate"] >= high_delay) |
                (customer["complaint_rate"] >= high_complaint)
            )
        ),

        # Loyal / Satisfied:
        # good experience + relatively low delays
        (
            (customer["avg_csat"] >= high_csat) &
            (customer["delay_rate"] <= low_delay)
        ),

        # High-Service-Need:
        # frequent service visits
        (
            customer["total_visits"] >= high_visits
        )
    ],

    [
        "At-Risk",
        "Loyal / Satisfied",
        "High-Service-Need"
    ],

    default="Occasional / Low-Engagement"
)
# ============================================================
# 7. SAVE CUSTOMER-LEVEL DATA
# ============================================================

customer.to_csv(
    f"{REPORT_PATH}/customer_segments.csv",
    index=False
)


# ============================================================
# 8. SEGMENT PROFILE
# ============================================================

segment_profile = (
    customer
    .groupby("segment")
    .agg(
        customers=("customer_id", "count"),
        avg_visits=("total_visits", "mean"),
        avg_delay_hours=("avg_delay_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        avg_csat=("avg_csat", "mean"),
        complaint_rate=("complaint_rate", "mean")
    )
    .reset_index()
)

segment_profile["customer_share"] = (
    segment_profile["customers"] /
    segment_profile["customers"].sum()
)

segment_profile.to_csv(
    f"{REPORT_PATH}/segment_profile.csv",
    index=False
)


# ============================================================
# 9. BUSINESS ACTIONS
# ============================================================

actions = pd.DataFrame({

    "segment": [
        "Loyal / Satisfied",
        "At-Risk",
        "High-Service-Need",
        "Occasional / Low-Engagement"
    ],

    "business_strategy": [
        "Maintain service quality and encourage retention",
        "Prioritize proactive communication and service recovery",
        "Focus on appointment efficiency and preventive service planning",
        "Improve engagement and provide relevant service reminders"
    ]
})

segment_profile = segment_profile.merge(
    actions,
    on="segment",
    how="left"
)

segment_profile.to_csv(
    f"{REPORT_PATH}/segment_profile_with_actions.csv",
    index=False
)


# ============================================================
# 10. IDENTIFY HIGH-PRIORITY CUSTOMERS
# ============================================================

at_risk = customer[
    customer["segment"] == "At-Risk"
].copy()

at_risk = at_risk.sort_values(
    by=[
        "complaint_rate",
        "delay_rate",
        "avg_csat"
    ],
    ascending=[
        False,
        False,
        True
    ]
)

at_risk.to_csv(
    f"{REPORT_PATH}/at_risk_customers.csv",
    index=False
)


# ============================================================
# 11. SEGMENT SUMMARY
# ============================================================

print("\n======================================")
print("CUSTOMER SEGMENT SUMMARY")
print("======================================")

print(
    segment_profile[
        [
            "segment",
            "customers",
            "customer_share",
            "avg_visits",
            "delay_rate",
            "avg_csat",
            "complaint_rate"
        ]
    ]
    .sort_values(
        "customers",
        ascending=False
    )
)


# ============================================================
# 12. KEY BUSINESS FINDINGS
# ============================================================

largest_segment = (
    segment_profile
    .sort_values("customers", ascending=False)
    .iloc[0]
)

highest_risk = (
    segment_profile
    .sort_values("complaint_rate", ascending=False)
    .iloc[0]
)

highest_delay = (
    segment_profile
    .sort_values("delay_rate", ascending=False)
    .iloc[0]
)

print("\nKEY FINDINGS")

print(
    f"\nLargest segment: "
    f"{largest_segment['segment']}"
)

print(
    f"Highest complaint segment: "
    f"{highest_risk['segment']}"
)

print(
    f"Highest delay-rate segment: "
    f"{highest_delay['segment']}"
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("PHASE 6 COMPLETED")
print("======================================")

print(
    f"\nReports saved to: {REPORT_PATH}"
)