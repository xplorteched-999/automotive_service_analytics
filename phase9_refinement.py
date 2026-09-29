import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase9"

RANDOM_STATE = 42

# ============================================================
# 1. LOAD EXISTING CUSTOMER FEATURES
# ============================================================

df = pd.read_csv(
    f"{REPORT_PATH}/customer_segments.csv"
)

features = [
    "visit_count",
    "unique_vehicles",
    "avg_delay_hours",
    "delay_rate",
    "avg_csat",
    "complaint_rate",
    "service_type_count",
    "recency_days"
]

X = df[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 2. COMPARE K=3 AND K=4
# ============================================================

results = []

for k in [3, 4]:

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

    results.append({
        "k": k,
        "silhouette_score": score
    })

validation = pd.DataFrame(results)

validation.to_csv(
    f"{REPORT_PATH}/k3_vs_k4_validation.csv",
    index=False
)

print("\nK COMPARISON")
print(validation)


# ============================================================
# 3. USE K=4 FOR BUSINESS INTERPRETABILITY
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster_id_refined"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# 4. PROFILE THE REFINED CLUSTERS
# ============================================================

profile = (
    df
    .groupby("cluster_id_refined")
    .agg(
        customers=("customer_id", "count"),
        avg_visits=("visit_count", "mean"),
        avg_vehicles=("unique_vehicles", "mean"),
        avg_delay_hours=("avg_delay_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        avg_csat=("avg_csat", "mean"),
        complaint_rate=("complaint_rate", "mean"),
        avg_service_types=("service_type_count", "mean"),
        avg_recency_days=("recency_days", "mean")
    )
    .reset_index()
)

profile["customer_percentage"] = (
    profile["customers"] /
    profile["customers"].sum()
) * 100


# ============================================================
# 5. CREATE DATA-DRIVEN SEGMENT NAMES
# ============================================================

# Highest satisfaction + lowest delay
satisfied_cluster = profile.sort_values(
    ["avg_csat", "delay_rate"],
    ascending=[False, True]
).iloc[0]["cluster_id_refined"]


# Highest delay + lowest satisfaction
risk_cluster = profile.sort_values(
    ["delay_rate", "avg_csat"],
    ascending=[False, True]
).iloc[0]["cluster_id_refined"]


# Highest engagement
engagement_cluster = profile[
    profile["cluster_id_refined"] != satisfied_cluster
].sort_values(
    "avg_visits",
    ascending=False
).iloc[0]["cluster_id_refined"]


# Remaining cluster
remaining_clusters = [
    x for x in profile["cluster_id_refined"]
    if x not in [
        satisfied_cluster,
        risk_cluster,
        engagement_cluster
    ]
]

other_cluster = remaining_clusters[0]


name_map = {
    satisfied_cluster: "Satisfied / Stable Customers",
    risk_cluster: "High-Risk Delay Customers",
    engagement_cluster: "Highly Engaged Customers",
    other_cluster: "At-Risk Experience Customers"
}

profile["segment_name"] = (
    profile["cluster_id_refined"]
    .map(name_map)
)


# ============================================================
# 6. BUSINESS ACTIONS
# ============================================================

action_map = {

    "Satisfied / Stable Customers":
        "Maintain service quality and encourage retention and advocacy.",

    "High-Risk Delay Customers":
        "Prioritize proactive delay alerts, appointment monitoring, and service recovery.",

    "Highly Engaged Customers":
        "Use personalized maintenance reminders and loyalty-oriented service communications.",

    "At-Risk Experience Customers":
        "Target complaint prevention, service education, and experience improvement."
}

profile["recommended_action"] = (
    profile["segment_name"]
    .map(action_map)
)


# ============================================================
# 7. SAVE REFINED CUSTOMER DATA
# ============================================================

df = df.merge(
    profile[
        [
            "cluster_id_refined",
            "segment_name"
        ]
    ],
    on="cluster_id_refined",
    how="left"
)

df.to_csv(
    f"{REPORT_PATH}/customer_segments_refined.csv",
    index=False
)

profile.to_csv(
    f"{REPORT_PATH}/segment_profiles_refined.csv",
    index=False
)


# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("REFINED SEGMENTATION")
print("======================================")

print(
    profile[
        [
            "cluster_id_refined",
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
print("PHASE 9 REFINEMENT COMPLETED")
print("======================================")