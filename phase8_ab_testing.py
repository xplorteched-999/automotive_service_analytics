import os
import pandas as pd
import numpy as np

from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower


# ============================================================
# 1. SETUP
# ============================================================

DATA_PATH = "data/processed"
PHASE7_PATH = "reports/phase7"
REPORT_PATH = "reports/phase8"

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

surveys = pd.read_csv(
    f"{DATA_PATH}/customer_surveys.csv"
)

risk_scores = pd.read_csv(
    f"{PHASE7_PATH}/scored_service_visits.csv"
)

print("Data loaded")

print("Visits:", visits.shape)
print("Appointments:", appointments.shape)
print("Surveys:", surveys.shape)
print("Risk scores:", risk_scores.shape)


# ============================================================
# 3. BUILD EXPERIMENT DATASET
# ============================================================

df = visits.merge(
    appointments[
        [
            "appointment_id",
            "customer_id",
            "vehicle_id",
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


# ============================================================
# 4. ADD MODEL RISK SCORE
# ============================================================

# Phase 7 scoring file contains the test-set observations.
risk_columns = [
    "dealer_id",
    "service_type",
    "lead_time_days",
    "scheduled_duration_hours",
    "parts_delayed",
    "parts_delay_days",
    "delay_probability",
    "risk_band"
]

available_risk_columns = [
    c for c in risk_columns
    if c in risk_scores.columns
]

risk_scores = risk_scores[
    available_risk_columns
].copy()

# Add unique row identifier to both datasets
# based on the modeling variables.
df["experiment_row_id"] = np.arange(
    len(df)
)

risk_scores["risk_row_id"] = np.arange(
    len(risk_scores)
)

# Since the Phase 7 scoring file contains only
# the test observations, we'll use the risk file
# as the experiment population.

experiment = risk_scores.copy()

print(
    "\nPotential experiment population:",
    len(experiment)
)


# ============================================================
# 5. CREATE ELIGIBLE POPULATION
# ============================================================

# We target HIGH-RISK visits because the business
# intervention is intended for visits where
# proactive communication is most useful.

experiment = experiment[
    experiment["risk_band"] == "High Risk"
].copy()

print(
    "High-risk eligible population:",
    len(experiment)
)


# ============================================================
# 6. RANDOMIZATION
# ============================================================

np.random.seed(RANDOM_STATE)

experiment["random_number"] = np.random.random(
    len(experiment)
)

experiment["experiment_group"] = np.where(
    experiment["random_number"] < 0.50,
    "Control",
    "Treatment"
)

print("\nExperiment allocation:")

print(
    experiment["experiment_group"]
    .value_counts()
)


# ============================================================
# 7. ATTACH OUTCOME DATA
# ============================================================

# Match experimental observations to actual visit outcomes.

outcomes = visits[
    [
        "dealer_id",
        "service_type",
        "delay_hours",
        "delay_flag"
    ]
].copy()

# The synthetic dataset can contain repeated combinations,
# so we use the available outcome distribution to construct
# a simulated experiment outcome below.

np.random.seed(RANDOM_STATE)


# ============================================================
# 8. SIMULATE INTERVENTION EFFECT
# ============================================================

# IMPORTANT:
#
# This is NOT a real experiment.
#
# We are simulating a plausible treatment effect
# because the historical dataset did not contain
# an actual treatment/control assignment.
#
# Treatment assumptions:
# - modest improvement in delay probability
# - modest improvement in CSAT
# - lower complaint probability
#
# These assumptions are clearly documented rather
# than presented as real business results.


# Base outcomes sampled from historical high-risk visits

high_risk_visits = visits[
    visits["delay_hours"] > 0
].copy()

sample_size = len(experiment)

sampled_outcomes = high_risk_visits.sample(
    n=sample_size,
    replace=True,
    random_state=RANDOM_STATE
).reset_index(drop=True)


experiment = experiment.reset_index(
    drop=True
)

experiment["delay_hours"] = (
    sampled_outcomes["delay_hours"]
)

experiment["baseline_delay_flag"] = (
    sampled_outcomes["delay_hours"] > 0
).astype(int)


# ------------------------------------------------------------
# Treatment effect
# ------------------------------------------------------------

np.random.seed(RANDOM_STATE)

# Treatment reduces probability of delay
# by a modest simulated effect.

treatment_effect = (
    (experiment["experiment_group"] == "Treatment") &
    (np.random.random(len(experiment)) < 0.10)
)

experiment["final_delay_flag"] = (
    experiment["baseline_delay_flag"]
)

experiment.loc[
    treatment_effect,
    "final_delay_flag"
] = 0


# ============================================================
# 9. SIMULATE CSAT
# ============================================================

# Use a realistic relationship between delay and CSAT.

np.random.seed(RANDOM_STATE)

base_csat = np.random.normal(
    loc=3.8,
    scale=0.8,
    size=len(experiment)
)

# Delay negatively affects satisfaction
base_csat -= (
    experiment["final_delay_flag"] * 0.7
)

# Treatment provides a small satisfaction improvement
base_csat += (
    (experiment["experiment_group"] == "Treatment")
    * 0.20
)

experiment["csat_score"] = np.clip(
    base_csat,
    1,
    5
)


# ============================================================
# 10. SIMULATE COMPLAINT OUTCOME
# ============================================================

np.random.seed(RANDOM_STATE)

complaint_probability = (
    0.08 +
    experiment["final_delay_flag"] * 0.20
)

# Treatment slightly reduces complaint probability
complaint_probability -= (
    (experiment["experiment_group"] == "Treatment")
    * 0.05
)

complaint_probability = np.clip(
    complaint_probability,
    0,
    1
)

experiment["complaint_flag"] = (
    np.random.random(len(experiment))
    < complaint_probability
).astype(int)


# ============================================================
# 11. EXPERIMENT SUMMARY
# ============================================================

group_summary = (
    experiment
    .groupby("experiment_group")
    .agg(
        customers=("experiment_group", "count"),

        delay_rate=(
            "final_delay_flag",
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

group_summary.to_csv(
    f"{REPORT_PATH}/experiment_group_summary.csv",
    index=False
)

print("\nEXPERIMENT SUMMARY")
print(group_summary)


# ============================================================
# 12. PRIMARY METRIC — CSAT
# ============================================================

control_csat = experiment.loc[
    experiment["experiment_group"] == "Control",
    "csat_score"
].dropna()

treatment_csat = experiment.loc[
    experiment["experiment_group"] == "Treatment",
    "csat_score"
].dropna()


# Mann-Whitney U is appropriate because CSAT
# is ordinal and bounded from 1 to 5.

u_stat, p_value_csat = mannwhitneyu(
    control_csat,
    treatment_csat,
    alternative="two-sided"
)

csat_difference = (
    treatment_csat.mean()
    - control_csat.mean()
)

csat_effect_percent = (
    csat_difference /
    control_csat.mean()
) * 100


test_csat = pd.DataFrame({
    "metric": ["CSAT"],
    "control_mean": [
        control_csat.mean()
    ],
    "treatment_mean": [
        treatment_csat.mean()
    ],
    "absolute_difference": [
        csat_difference
    ],
    "relative_difference_percent": [
        csat_effect_percent
    ],
    "u_statistic": [
        u_stat
    ],
    "p_value": [
        p_value_csat
    ],
    "significant_at_0_05": [
        p_value_csat < 0.05
    ]
})

test_csat.to_csv(
    f"{REPORT_PATH}/test_primary_csat.csv",
    index=False
)

print("\nPRIMARY TEST — CSAT")
print(test_csat)


# ============================================================
# 13. SECONDARY METRIC — DELAY RATE
# ============================================================

control_delay = experiment.loc[
    experiment["experiment_group"] == "Control",
    "final_delay_flag"
]

treatment_delay = experiment.loc[
    experiment["experiment_group"] == "Treatment",
    "final_delay_flag"
]

delay_counts = np.array([
    treatment_delay.sum(),
    control_delay.sum()
])

delay_nobs = np.array([
    len(treatment_delay),
    len(control_delay)
])

z_stat, p_value_delay = proportions_ztest(
    delay_counts,
    delay_nobs
)

delay_rate_difference = (
    treatment_delay.mean()
    - control_delay.mean()
)

delay_relative_change = (
    delay_rate_difference /
    control_delay.mean()
) * 100


test_delay = pd.DataFrame({
    "metric": ["Delay Rate"],
    "control_rate": [
        control_delay.mean()
    ],
    "treatment_rate": [
        treatment_delay.mean()
    ],
    "absolute_difference": [
        delay_rate_difference
    ],
    "relative_change_percent": [
        delay_relative_change
    ],
    "z_statistic": [
        z_stat
    ],
    "p_value": [
        p_value_delay
    ],
    "significant_at_0_05": [
        p_value_delay < 0.05
    ]
})

test_delay.to_csv(
    f"{REPORT_PATH}/test_secondary_delay.csv",
    index=False
)

print("\nSECONDARY TEST — DELAY RATE")
print(test_delay)


# ============================================================
# 14. SECONDARY METRIC — COMPLAINT RATE
# ============================================================

control_complaints = experiment.loc[
    experiment["experiment_group"] == "Control",
    "complaint_flag"
]

treatment_complaints = experiment.loc[
    experiment["experiment_group"] == "Treatment",
    "complaint_flag"
]

complaint_table = pd.crosstab(
    experiment["experiment_group"],
    experiment["complaint_flag"]
)

chi2, p_value_complaint, dof, expected = (
    chi2_contingency(
        complaint_table
    )
)

complaint_difference = (
    treatment_complaints.mean()
    - control_complaints.mean()
)

test_complaint = pd.DataFrame({
    "metric": ["Complaint Rate"],
    "control_rate": [
        control_complaints.mean()
    ],
    "treatment_rate": [
        treatment_complaints.mean()
    ],
    "absolute_difference": [
        complaint_difference
    ],
    "chi_square": [
        chi2
    ],
    "p_value": [
        p_value_complaint
    ],
    "significant_at_0_05": [
        p_value_complaint < 0.05
    ]
})

test_complaint.to_csv(
    f"{REPORT_PATH}/test_secondary_complaint.csv",
    index=False
)

print("\nSECONDARY TEST — COMPLAINT RATE")
print(test_complaint)


# ============================================================
# 15. EFFECT SIZE
# ============================================================

pooled_std = np.sqrt(
    (
        (len(control_csat) - 1)
        * control_csat.var()
        +
        (len(treatment_csat) - 1)
        * treatment_csat.var()
    )
    /
    (
        len(control_csat)
        +
        len(treatment_csat)
        - 2
    )
)

cohens_d = (
    treatment_csat.mean()
    - control_csat.mean()
) / pooled_std


effect_size = pd.DataFrame({
    "metric": ["CSAT"],
    "cohens_d": [cohens_d]
})

effect_size.to_csv(
    f"{REPORT_PATH}/effect_size.csv",
    index=False
)

print("\nEFFECT SIZE")
print(effect_size)


# ============================================================
# 16. SAMPLE SIZE / POWER CONSIDERATION
# ============================================================

observed_effect = abs(
    cohens_d
)

power_analysis = TTestIndPower()

required_sample_size = (
    power_analysis.solve_power(
        effect_size=max(
            observed_effect,
            0.01
        ),
        alpha=0.05,
        power=0.80,
        ratio=1.0,
        alternative="two-sided"
    )
)

sample_size_report = pd.DataFrame({
    "parameter": [
        "Observed Cohen's d",
        "Alpha",
        "Desired Power",
        "Required Sample Size Per Group",
        "Actual Control Size",
        "Actual Treatment Size"
    ],
    "value": [
        observed_effect,
        0.05,
        0.80,
        required_sample_size,
        len(control_csat),
        len(treatment_csat)
    ]
})

sample_size_report.to_csv(
    f"{REPORT_PATH}/sample_size_analysis.csv",
    index=False
)

print("\nSAMPLE SIZE ANALYSIS")
print(sample_size_report)


# ============================================================
# 17. EXPERIMENT DECISION
# ============================================================

if p_value_csat < 0.05:

    primary_result = (
        "Statistically significant difference "
        "in CSAT between treatment and control."
    )

else:

    primary_result = (
        "No statistically significant difference "
        "in CSAT was detected."
    )


if (
    p_value_delay < 0.05
):

    delay_result = (
        "Statistically significant difference "
        "in delay rate."
    )

else:

    delay_result = (
        "No statistically significant difference "
        "in delay rate was detected."
    )


if (
    p_value_complaint < 0.05
):

    complaint_result = (
        "Statistically significant difference "
        "in complaint rate."
    )

else:

    complaint_result = (
        "No statistically significant difference "
        "in complaint rate was detected."
    )


decision_report = pd.DataFrame({
    "metric": [
        "Primary — CSAT",
        "Secondary — Delay",
        "Secondary — Complaint"
    ],
    "result": [
        primary_result,
        delay_result,
        complaint_result
    ],
    "p_value": [
        p_value_csat,
        p_value_delay,
        p_value_complaint
    ]
})

decision_report.to_csv(
    f"{REPORT_PATH}/experiment_decision.csv",
    index=False
)


# ============================================================
# 18. FINAL EXPERIMENT REPORT
# ============================================================

print("\n======================================")
print("A/B TEST RESULT")
print("======================================")

print("\nPrimary metric:")
print(primary_result)

print("\nDelay metric:")
print(delay_result)

print("\nComplaint metric:")
print(complaint_result)

print("\nIMPORTANT:")
print(
    "This experiment is SIMULATED because "
    "the source dataset does not contain a "
    "real randomized treatment/control assignment."
)


# ============================================================
# 19. SAVE EXPERIMENT DATA
# ============================================================

experiment.to_csv(
    f"{REPORT_PATH}/experiment_dataset.csv",
    index=False
)


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("PHASE 8 COMPLETED")
print("======================================")

print(
    f"\nReports saved to: {REPORT_PATH}"
)