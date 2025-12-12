Credit Risk Probability Model Using Alternative Data

This project implements an end-to-end Credit Risk Scoring System for Bati Bank, using eCommerce transaction data.
The solution includes data processing, proxy target creation, model training, experiment tracking, and API deployment.

## 1. Project Overview

Bati Bank is partnering with an eCommerce company to launch a Buy-Now-Pay-Later (BNPL) service.
To safely approve credit, the bank needs a model that predicts which customers are high-risk (may default) vs. low-risk.

However, the dataset does not contain a real default label.
Therefore, this challenge involves creating a proxy risk label based on customer behavior and using it to train ML models.

## 1. Project Overview

credit-risk-model/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── predict.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
│   └── test_data_processing.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## 3. Credit Scoring Business Understanding (Task 1)
### 3.1 Basel II and the Need for Interpretable, Well-Documented Models

The Basel II Accord requires banks to use transparent and explainable risk-measurement systems because credit risk directly impacts capital allocation and regulatory compliance.
Banks must demonstrate:

how the model works,

why decisions are made,

and whether the model is stable and reliable.

For this reason, models such as Logistic Regression with Weight of Evidence (WoE) are often preferred in credit scoring—they provide clear, interpretable contributions of each feature.
This project therefore prioritizes documentation, explainability, and reproducibility.

### 3.2 Why a Proxy Target Variable Is Necessary and What Risks It Brings

The provided dataset does not include a “default” or “loan outcome” column.
Machine learning models require a target variable for supervised training, so we must create a proxy risk label.

Using customer behavior patterns (Recency, Frequency, Monetary value), we identify customers who appear least engaged and therefore more likely to be high-risk.
These customers are labeled as:

1 = high-risk

0 = low-risk

This approach enables model training, but it carries risks:

The proxy may not fully represent real default behavior

Some customers labeled “high-risk” may be safe borrowers

The model may learn patterns that do not reflect true credit performance

Incorrect predictions could lead to rejecting good borrowers or approving risky ones

This proxy label is therefore treated as a temporary, behavior-based approximation, not a replacement for real repayment data.

### 3.3 Trade-offs: Simple vs. Complex Models in Regulated Environments
Model Type	                                 Benefits	                                                           Limitations

Logistic Regression + WoE	 Highly interpretable, stable, regulator-friendly, easy to audit  	Lower accuracy, cannot capture non-linear patterns

Gradient Boosting, XGBoost, 
Random Forest	             Higher predictive power, captures complex interactions	            Harder to explain, less transparent

Banks often prefer simpler models because they prioritize interpretability and compliance over maximum accuracy.

## 4. Tasks Summary
- Task 1: Credit Scoring Business Understanding ✔
- Task 2: Exploratory Data Analysis (EDA)
- Task 3: Feature Engineering Pipeline
- Task 4: Proxy Risk Label Creation (RFM + Clustering)
- Task 5: Model Training + MLflow Tracking
- Task 6: API Deployment + Docker + CI/CD
## 5. Installation and Environment Setup

Follow the steps below to set up your environment.
⭐ HOW TO SET UP THE ENVIRONMENT (Windows PowerShell)

Go to your project folder:
PS D:\KAIM\weak-4\credit-risk-model>

STEP 1 — Create a Virtual Environment
python -m venv .venv

Activate it:
.venv\Scripts\activate

You should now see:
(.venv) PS D:\KAIM\weak-4\credit-risk-model>

✔ STEP 2 — Install Requirements
First open your requirements.txt and paste this:
pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow
pytest
fastapi
uvicorn
xgboost
lightgbm
xverse
pyarrow

Then install:
pip install -r requirements.txt

Make sure packages appear in the list.