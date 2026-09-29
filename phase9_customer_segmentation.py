import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. SETUP
# ============================================================

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase9"

os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 4


# ============================================================
# 2. LOAD DATA
# ============================================================

customers = pd.read_csv(
    f"{DATA_PATH}/customers.csv"
)

appointments = pd.read_csv(
    f"{DATA_PATH}/appointments.csv"
)

visits = pd.read_csv(
    f"{DATA_PATH}/service_visits.csv"
)

surveys = pd.read_csv(
    f"{DATA_PATH}/customer_surveys.csv"
)

print("======================================")
print("PHASE 9 — CUSTOMER SEGMENTATION")
print("======================================")

print("\nData loaded:")
print("Customers:", customers.shape)
print("Appointments:", appointments.shape)
print("Visits:", visits.shape)
print("Surveys:", surveys.shape)


# ============================================================
# 3. PREPARE DATES
# ============================================================

appointments["appointment_date"] = pd.to_datetime(
    appointments["appointment_date"],
    errors="coerce"
)

visits["check_in_time"] = pd.to_datetime(
    visits["check_in_time"],
    errors="coerce"
)


# ============================================================
# 4. MERGE VISITS + APPOINTMENTS
# ============================================================

# service_type already exists in service_visits,
# so we don't need to bring it from appointments.

visit_customer = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "vehicle_id"
        ]
    ],
    on="appointment_id",
    how="left"
)

print("\nCustomer-visit dataset:")
print(visit_customer.shape)

print("\nColumns:")
print(visit_customer.columns.tolist())


# ============================================================
# 5. MERGE SURVEYS
# ============================================================

visit_customer = visit_customer.merge(
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

print("\nAfter survey merge:")
print(visit_customer.shape)


# ============================================================
# 6. CUSTOMER-LEVEL FEATURES
# ============================================================

customer_features = (
    visit_customer
    .groupby("customer_id")
    .agg(
        visit_count=("visit_id", "count"),

        unique_vehicles=("vehicle_id", "nunique"),

        avg_delay_hours=("delay_hours", "mean"),

        delay_rate=("delay_flag", "mean"),

        avg_csat=("csat_score", "mean"),

        complaint_rate=("complaint_flag", "mean"),

        avg_service_duration=("actual_duration_hours", "mean"),

        service_type_count=("service_type", "nunique"),

        last_visit_date=("check_in_time", "max")
    )
    .reset_index()
)


# ============================================================
# 7. RECENCY
# ============================================================

reference_date = visit_customer["check_in_time"].max()

customer_features["recency_days"] = (
    reference_date -
    customer_features["last_visit_date"]
).dt.days

customer_features["recency_days"] = (
    customer_features["recency_days"]
    .fillna(
        customer_features["recency_days"].median()
    )
)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

customer_features["avg_csat"] = (
    customer_features["avg_csat"]
    .fillna(
        customer_features["avg_csat"].median()
    )
)

customer_features["complaint_rate"] = (
    customer_features["complaint_rate"]
    .fillna(0)
)


print("\nCustomer feature dataset:")
print(customer_features.shape)


# ============================================================
# 9. CLUSTERING FEATURES
# ============================================================

cluster_features = [
    "visit_count",
    "unique_vehicles",
    "avg_delay_hours",
    "delay_rate",
    "service_type_count",
    "recency_days"
]
X = customer_features[
    cluster_features
].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median()
)


# ============================================================
# 10. STANDARDIZE
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 11. TEST CLUSTER COUNTS
# ============================================================

print("\n======================================")
print("CLUSTER VALIDATION")
print("======================================")

silhouette_results = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append({
        "k": k,
        "silhouette_score": score
    })

    print(
        f"k={k} | "
        f"Silhouette Score={score:.4f}"
    )

silhouette_df = pd.DataFrame(
    silhouette_results
)

silhouette_df.to_csv(
    f"{REPORT_PATH}/silhouette_scores.csv",
    index=False
)

best_k = int(
    silhouette_df.loc[
        silhouette_df["silhouette_score"].idxmax(),
        "k"
    ]
)

print(
    f"\nBest statistical k: {best_k}"
)

print(
    f"Business k selected: {N_CLUSTERS}"
)


# ============================================================
# 12. FINAL K-MEANS
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

customer_features["cluster_id"] = (
    kmeans.fit_predict(X_scaled)
)


# ============================================================
# 13. PROFILE SEGMENTS
# ============================================================

profile = (
    customer_features
    .groupby("cluster_id")
    .agg(
        customers=("customer_id", "count"),

        avg_visits=("visit_count", "mean"),

        avg_unique_vehicles=(
            "unique_vehicles",
            "mean"
        ),

        avg_delay_hours=(
            "avg_delay_hours",
            "mean"
        ),

        delay_rate=(
            "delay_rate",
            "mean"
        ),

        avg_csat=(
            "avg_csat",
            "mean"
        ),

        complaint_rate=(
            "complaint_rate",
            "mean"
        ),

        avg_service_types=(
            "service_type_count",
            "mean"
        ),

        avg_recency_days=(
            "recency_days",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# 14. CUSTOMER PERCENTAGE
# ============================================================

total_customers = len(
    customer_features
)

profile["customer_percentage"] = (
    profile["customers"] /
    total_customers
) * 100


# ============================================================
# 15. BUSINESS SEGMENT NAMES
# ============================================================

# We assign names based on the actual profiles.

high_delay_cluster = int(
    profile.loc[
        profile["delay_rate"].idxmax(),
        "cluster_id"
    ]
)

high_csat_cluster = int(
    profile.loc[
        profile["avg_csat"].idxmax(),
        "cluster_id"
    ]
)

high_visit_cluster = int(
    profile.loc[
        profile["avg_visits"].idxmax(),
        "cluster_id"
    ]
)

profile["segment_name"] = "Occasional Customers"

profile.loc[
    profile["cluster_id"] == high_delay_cluster,
    "segment_name"
] = "Delay-Sensitive Customers"

# Only assign this if it isn't already the delay segment
if high_csat_cluster != high_delay_cluster:

    profile.loc[
        profile["cluster_id"] == high_csat_cluster,
        "segment_name"
    ] = "Satisfied Loyal Customers"


if (
    high_visit_cluster != high_delay_cluster
    and
    high_visit_cluster != high_csat_cluster
):

    profile.loc[
        profile["cluster_id"] == high_visit_cluster,
        "segment_name"
    ] = "High-Engagement Customers"


# ============================================================
# 16. SEGMENT MAP
# ============================================================

segment_map = profile[
    [
        "cluster_id",
        "segment_name"
    ]
]

customer_segments = customer_features.merge(
    segment_map,
    on="cluster_id",
    how="left"
)


# ============================================================
# 17. RECOMMENDATIONS
# ============================================================

recommendation_map = {

    "Delay-Sensitive Customers":
        "Prioritize proactive delay communication, "
        "appointment monitoring, and recovery outreach.",

    "Satisfied Loyal Customers":
        "Maintain service quality and encourage "
        "retention, repeat service, and advocacy.",

    "High-Engagement Customers":
        "Use personalized service reminders, "
        "maintenance recommendations, and loyalty messaging.",

    "Occasional Customers":
        "Use targeted reminders and service education "
        "to increase repeat engagement."
}

profile["recommended_action"] = (
    profile["segment_name"]
    .map(recommendation_map)
)


# ============================================================
# 18. SAVE OUTPUTS
# ============================================================

customer_segments.to_csv(
    f"{REPORT_PATH}/customer_segments.csv",
    index=False
)

profile.to_csv(
    f"{REPORT_PATH}/segment_profiles.csv",
    index=False
)

silhouette_df.to_csv(
    f"{REPORT_PATH}/cluster_validation.csv",
    index=False
)

profile.to_csv(
    f"{REPORT_PATH}/customer_segmentation_report.csv",
    index=False
)


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("CUSTOMER SEGMENTATION RESULTS")
print("======================================")

print(
    profile[
        [
            "cluster_id",
            "segment_name",
            "customers",
            "customer_percentage",
            "avg_visits",
            "delay_rate",
            "avg_csat",
            "complaint_rate",
            "recommended_action"
        ]
    ].to_string(index=False)
)

print("\n======================================")
print("PHASE 9 COMPLETED")
print("======================================")

print(
    f"\nReports saved to: {REPORT_PATH}"
)