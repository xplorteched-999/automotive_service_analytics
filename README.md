# Automotive Service Operations Analytics

## Overview

An end-to-end analytics project focused on improving automotive service operations, customer experience, and proactive service-delay management.

The project combines data quality analysis, SQL, statistical testing, customer segmentation, predictive modeling, business reporting, and cloud architecture.

> **Note:** This project uses synthetic data created for portfolio and analytical demonstration purposes. Results should not be interpreted as actual automotive-company performance.

---

## Business Problem

Service organizations need to understand:

* Why service visits are delayed
* Which appointments are at higher risk of delay
* How service delays affect customer experience
* Which customer groups require different interventions
* How analytics can support proactive operational decisions

### Objective

Build an end-to-end analytics solution that transforms service-operation data into actionable business insights.

---

## Project Workflow

**Data Generation → Data Quality → SQL Analysis → Statistical Analysis → Customer Segmentation → Predictive Modeling → A/B Testing → Power BI → GCP Architecture → Business Recommendations**

---

## Key Results

### Predictive Model

The project developed a classification model to predict whether a service visit would be delayed.

| Metric    | Result |
| --------- | -----: |
| ROC-AUC   |  0.852 |
| Precision |  0.930 |
| Recall    |  0.741 |
| F1 Score  |  0.825 |

The model was used to classify service visits into low-, medium-, and high-risk groups for potential proactive intervention.

### Customer Segmentation

Customer behavior was analyzed using clustering and business-oriented segment interpretation.

Four refined segments were identified:

* Highly Engaged Customers
* High-Risk Delay Customers
* Satisfied / Stable Customers
* At-Risk Experience Customers

### A/B Testing

A simulated experiment was used to evaluate the relationship between customer communication and service outcomes.

The analysis found statistically significant differences in CSAT and complaint rate, while the difference in delay rate was not statistically significant.

> The experiment is simulated and does not represent a production randomized controlled trial.

---

## Data Quality

The project includes automated data-quality and relationship checks covering:

* Missing values
* Duplicate records
* Invalid relationships
* Referential integrity
* Data-type consistency
* Business-rule validation

Post-cleaning relationship checks achieved approximately 100% integrity across the validated relationships, with the documented exception in the repair-to-service-visit relationship.

---

## Technology Stack

### Programming & Analytics

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy

### Database & Querying

* SQL
* Relational data modeling

### Visualization & Reporting

* Power BI
* Python visualizations
* Executive business reporting

### Cloud Architecture

* Google Cloud Storage
* Google BigQuery
* GCP-based analytical architecture

---

## Power BI

The Power BI solution is designed to provide operational and customer-experience monitoring.

Key KPIs include:

* Total Visits
* Delayed Visits
* Delay Rate
* Average Delay Hours
* Average CSAT
* Complaint Rate
* First-Time-Fix Rate
* Unique Customers
* Visits per Customer
* Severe Delays

The dashboard is designed to support operational monitoring and management decision-making.

---

## GCP Architecture

The proposed cloud architecture separates raw data storage from analytical processing.

```text
Synthetic / Operational Data
            |
            v
   Google Cloud Storage
            |
            v
       BigQuery Raw
            |
            v
   BigQuery Analytics
            |
       +----+----+
       |         |
       v         v
    SQL       Python
       |         |
       +----+----+
            |
            v
    Predictive Analytics
            |
            v
        Power BI
            |
            v
   Business Decisions
```

The architecture is designed for scalability, data-quality monitoring, analytical processing, and future production deployment.

---

## Business Recommendations

### 1. Prioritize High-Risk Visits

Use predicted delay probability to identify appointments that may require operational attention.

### 2. Improve Proactive Communication

Provide customers with timely communication when a service visit has elevated delay risk.

### 3. Focus on At-Risk Customer Experience

Combine delay, CSAT, and complaint indicators to identify customers who may benefit from targeted service recovery.

### 4. Monitor Dealer Performance

Track operational and customer-experience KPIs by dealer to identify locations requiring further investigation.

---

## Project Structure

```text
automotive-service-analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── reports/
│   ├── phase7_model_results/
│   ├── phase8_ab_test/
│   ├── phase9_segmentation/
│   └── phase10_powerbi/
│
├── architecture/
│   └── gcp_architecture.md
│
├── README.md
└── requirements.txt
```

---

## Limitations

This is a portfolio project based on synthetic data.

Important limitations include:

* Results do not represent actual company performance.
* The A/B test is simulated.
* Customer segments require validation using real customer behavior.
* Model performance should be re-evaluated using production data.
* Production deployment would require monitoring for data drift, model drift, and data-quality issues.
* Business impact should be validated through controlled production pilots.

---

## Future Improvements

Potential future enhancements include:

* Deploying the analytical pipeline on GCP
* Connecting BigQuery directly to Power BI
* Automated data-quality monitoring
* Model monitoring and retraining
* Real-time or near-real-time risk scoring
* More advanced customer-experience modeling
* Production experimentation framework

---

## Author

Portfolio project demonstrating an end-to-end approach to data analytics, customer analytics, predictive modeling, statistical analysis, and business decision support.


