import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. SETUP
# ============================================================

DATA_PATH = "data/processed"
REPORT_PATH = "reports/phase7"

os.makedirs(REPORT_PATH, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

visits = pd.read_csv(
    f"{DATA_PATH}/service_visits.csv"
)

appointments = pd.read_csv(
    f"{DATA_PATH}/appointments.csv"
)

repairs = pd.read_csv(
    f"{DATA_PATH}/repairs.csv"
)

parts = pd.read_csv(
    f"{DATA_PATH}/parts.csv"
)

print("Data loaded")
print("Visits:", visits.shape)
print("Appointments:", appointments.shape)
print("Repairs:", repairs.shape)
print("Parts:", parts.shape)


# ============================================================
## ============================================================
# 3. BUILD MODELING DATASET
# ============================================================

df = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "vehicle_id",
            "booking_date",
            "appointment_date",
            "scheduled_duration_hours"
        ]
    ],
    on="appointment_id",
    how="left"
)

# Convert dates
df["appointment_date"] = pd.to_datetime(
    df["appointment_date"],
    errors="coerce"
)

df["booking_date"] = pd.to_datetime(
    df["booking_date"],
    errors="coerce"
)

# Appointment lead time
df["lead_time_days"] = (
    df["appointment_date"] -
    df["booking_date"]
).dt.days


# ============================================================
# # ============================================================
# 4. CREATE PARTS-RELATED FEATURES
# ============================================================

parts["part_order_date"] = pd.to_datetime(
    parts["part_order_date"],
    errors="coerce"
)

parts["part_received_date"] = pd.to_datetime(
    parts["part_received_date"],
    errors="coerce"
)

parts["parts_delay_days"] = (
    parts["part_received_date"] -
    parts["part_order_date"]
).dt.days

parts["part_delayed"] = (
    parts["parts_delay_flag"] == 1
)

repair_parts = repairs[
    ["repair_id", "visit_id"]
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
        parts_delay_days=("parts_delay_days", "max"),
        parts_delayed=("part_delayed", "max")
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

df["parts_delay_days"] = (
    df["parts_delay_days"]
    .fillna(0)
)

# ============================================================
# 5. CREATE TARGET
# ============================================================

df["delayed_visit"] = (
    df["delay_hours"] > 0
).astype(int)

print("\nTarget distribution:")
print(
    df["delayed_visit"]
    .value_counts(normalize=True)
)


# ============================================================
# 6. SELECT PREDICTIVE FEATURES
# ============================================================

features = [
    "dealer_id",
    "service_type",
    "lead_time_days",
    "scheduled_duration_hours",
    "parts_delayed",
    "parts_delay_days"
]

target = "delayed_visit"

model_df = df[
    features + [target]
].copy()

model_df = model_df.dropna(
    subset=[target]
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 8. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "dealer_id",
    "service_type"
]

numeric_features = [
    "lead_time_days",
    "scheduled_duration_hours",
    "parts_delayed",
    "parts_delay_days"
]


# ============================================================
# 9. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully.")


# ============================================================
# 12. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

metrics = pd.DataFrame({
    "metric": [
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1"
    ],
    "value": [
        roc_auc,
        precision,
        recall,
        f1
    ]
})

metrics.to_csv(
    f"{REPORT_PATH}/model_metrics.csv",
    index=False
)

print("\nMODEL PERFORMANCE")
print(metrics)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

confusion = pd.DataFrame(
    cm,
    index=[
        "Actual On-Time",
        "Actual Delayed"
    ],
    columns=[
        "Predicted On-Time",
        "Predicted Delayed"
    ]
)

confusion.to_csv(
    f"{REPORT_PATH}/confusion_matrix.csv"
)

print("\nCONFUSION MATRIX")
print(confusion)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

classification_df = pd.DataFrame(
    report
).transpose()

classification_df.to_csv(
    f"{REPORT_PATH}/classification_report.csv"
)


# ============================================================
# 16. FEATURE COEFFICIENTS
# ============================================================

trained_preprocessor = (
    pipeline.named_steps["preprocessing"]
)

trained_model = (
    pipeline.named_steps["model"]
)

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

coefficients = (
    trained_model
    .coef_[0]
)

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(
        coefficients
    )
})

feature_importance = (
    feature_importance
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)

feature_importance.to_csv(
    f"{REPORT_PATH}/feature_coefficients.csv",
    index=False
)

print("\nTOP MODEL FEATURES")

print(
    feature_importance.head(15)
)


# ============================================================
# 17. CREATE RISK SCORES FOR TEST VISITS
# ============================================================

scored_test = X_test.copy()

scored_test["actual_delayed"] = (
    y_test.values
)

scored_test["delay_probability"] = (
    y_probability
)

scored_test["predicted_delayed"] = (
    y_pred
)

scored_test["risk_band"] = pd.cut(
    scored_test["delay_probability"],
    bins=[
        -0.01,
        0.30,
        0.60,
        1.00
    ],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

scored_test.to_csv(
    f"{REPORT_PATH}/scored_service_visits.csv",
    index=False
)


# ============================================================
# 18. RISK SUMMARY
# ============================================================

risk_summary = (
    scored_test
    .groupby("risk_band", observed=True)
    .agg(
        visits=("predicted_delayed", "count"),
        actual_delay_rate=("actual_delayed", "mean"),
        avg_predicted_probability=(
            "delay_probability",
            "mean"
        )
    )
    .reset_index()
)

risk_summary.to_csv(
    f"{REPORT_PATH}/risk_summary.csv",
    index=False
)

print("\nRISK SUMMARY")
print(risk_summary)


# ============================================================
# 19. BUSINESS INTERPRETATION
# ============================================================

print("\n======================================")
print("BUSINESS INTERPRETATION")
print("======================================")

print(
    f"\nROC-AUC: {roc_auc:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall: {recall:.3f}"
)

print(
    f"F1 Score: {f1:.3f}"
)

print(
    "\nThe model estimates the probability that "
    "a service visit will be delayed."
)

print(
    "The intended business use is to identify "
    "higher-risk visits early enough for proactive intervention."
)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("PHASE 7 COMPLETED")
print("======================================")

print(
    f"\nReports saved to: {REPORT_PATH}"
)