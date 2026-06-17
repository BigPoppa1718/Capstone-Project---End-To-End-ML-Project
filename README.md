# Phase 1: Data Engineering & Domain-Specific Preprocessing Pipeline

This directory handles the raw synthesis, structural cleaning, and mathematical preparation of the provider reimbursement dataset. The pipeline maps complex, real-world commercial healthcare billing mechanics into structured vectors optimized for machine learning algorithms.

## 🩺 Healthcare Domain Rules & Architectural Decisions

To ensure a production-ready system that matches modern insurance claim architectures, the pipeline enforces several structural constraints:

### 1. The "Lesser Of" Billing Enforcement
In real-world commercial health coverage, an insurer will never reimburse a provider more than what they explicitly billed, even if the pre-negotiated contract allowed rate is higher. 
* **Implementation**: The final reimbursement calculation applies a strict minimum ceiling between the computed `allowed_amount` and the `submitted_charges` (`min(allowed_amount, submitted_charges)`).
* **Impact**: This creates important non-linear boundaries in our underlying dataset. It ensures that the model learns that high-tier insurance plans cannot override an under-billed medical claim.

### 2. Service Weights (Relative Value Units)
Procedures are not valued equally. Rather than assigning random arbitrary costs, the data applies a `service_weight` modifier to simulate Relative Value Units (RVUs) or Case Mix Indices.
* **Implementation**: Standard evaluation and management visits (CPT codes 99213, 99214, 99204) are pegged at a baseline weight of `1.0`. High-complexity surgical procedures (CPT code 27447 - Total Knee Arthroplasty) scale to a weight of `8.5`.

### 3. Policy Ceiling Caps
To mimic strict risk management layers built into health plans, a hard global maximum policy cap of `$12,000.00` is enforced on all final payouts. 

---

## 🛠️ Data Preprocessing & Cleaning Choices

### 1. Domain-Agnostic Missing Value Imputation
Electronic Health Records (EHR) routinely present missing values due to administrative oversight or manual entry human errors.
* **Numerical Metrics (`submitted_charges`)**: Utilizes a **Median Imputation** strategy. Healthcare claim distributions contain extreme pricing spikes due to heavy surgeries. Standard mean imputations skew baseline metrics upward; a median value preserves realistic billing distributions.
* **Categorical Metrics (`facility_type`)**: Utilizes a **Mode Imputation** (most frequent class) to handle missing environmental descriptors cleanly without introducing non-existent labels.

### 2. Normalization & Feature Scaling
Because healthcare costs and service weights span completely different scales (e.g., standard weights of 1.0 vs. dollar figures in the thousands), a `StandardScaler` is applied to transform all numerical features to a mean of `0` and a variance of `1`. This normalization ensures rapid, stable optimization convergence across gradient-based and linear estimators.

### 3. One-Hot Encoding & The Dummy Trap Protection
To transform text parameters into machine-readable numeric arrays, a scikit-learn `OneHotEncoder` is initialized with `drop='first'`. Dropping the first dummy column prevents perfect multi-collinearity, which natively breaks core mathematical assumptions in algorithms like Logistic Regression.

---

## 🛡️ Data Leakage Prevention Matrix

To maintain production integrity, the data features are isolated into strict categories. When a user queries the LLM interface in a live production environment, the actual final payment fields do not exist yet. 

| Training Features (Allowed `X`) | Leakage Targets (Strictly Excluded) | Final Label (`y`) |
| :--- | :--- | :--- |
| `provider_specialty` | `allowed_amount` | **`exceeds_budget`** |
| `facility_type` | `final_reimbursement_rate` | |
| `cpt_code` | `percent_of_charge` | |
| `state` | `is_capped_by_charges` | |
| `insurer_tier` | `is_capped_by_policy` | |
| `service_weight` | | |
| `submitted_charges` | | |
| `target_budget` | | |


# Phase 2: Multi-Model Experimentation & Tracking Pipeline

This phase implements the training, logging, and evaluation infrastructure for Component 1 of the Capstone project. It establishes an enterprise configuration-driven framework to train and benchmark multiple machine learning architectures simultaneously while tracking all metadata using MLflow.

## 🏗️ Architecture Design Decisions

### 1. Decoupled, Configuration-Driven Framework
Hardcoded training scripts create significant operational friction and vulnerability during model updates. 
* **Implementation**: All dataset definitions, validation split rules, and algorithm-specific hyperparameters are decoupled from the code execution layer and isolated inside `config.yaml`.
* **Impact**: The training script (`train_experiments.py`) functions as a generic execution engine. System parameters can be tuned or entirely new model variables introduced without modifying a single line of Python source code.

### 2. Multi-Model Matrixing
To satisfy production validation standards, the script evaluates three distinctly different algorithm classes side-by-side:
* **Logistic Regression**: Serves as our linear baseline benchmark.
* **Random Forest Classifier**: An ensemble bagging approach designed to capture localized regional feature splits.
* **XGBoost (Gradient Boosting)**: An optimization boosting framework engineered to minimize residual error on highly dense structures.

### 3. Production Database Migration (SQLite Backend)
Modern MLflow instances restrict traditional raw local directory file-system tracking (`./mlruns`). To comply with enterprise data architecture standards and bypass environment blocks, this pipeline explicitly forces a structured relational backend store.
* **Implementation**: Embedded URI tracking via `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.
* **Impact**: Transactions, parameters, and evaluations are logged cleanly into a localized SQL database file (`mlflow.db`), enabling fast querying and future deployment scaling.

---

## 📈 Experiment Metrics & Analysis

The pipeline tracks structural performance matrix indicators on the holdout test validation subset:

| Algorithm Model Configuration | F1-Score | Recall (Overrun Detection) | Status |
| :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | 0.7177 | **0.7642** | High Overrun Capture |
| **Ensemble Random Forest** | 0.7260 | 0.7210 | Stable Baseline |
| **XGBoost Gradient Boosting** | **0.7416** | 0.7583 | **Production Selection Winner** |

### Core Operational Takeaways:
* **XGBoost** is selected as our core deployment engine for Component 2 due to its superior harmonic optimization balance (`F1-Score: 0.7416`). It maps non-linear billing conditions (such as specific regional multipliers and global payment caps) more efficiently than standard decision trees.
* **Logistic Regression** yielded a powerful fallback metric with a peak **Recall of 0.7642**. In healthcare financial management, catching a higher percentage of actual budget-exceeding procedures minimizes catastrophic financial blindspots.

---

## 🛠️ MLOps Tracking Dashboard Integration

Every automated execution run captures and streams data directly to your local SQL relational store.

### Launching the Dashboard Interface:
Execute this command in your root repository terminal folder to launch the web-based visualization server:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` in your web browser to access the tracking interface. Select the experiment named **`Provider_Reimbursement_Prediction`** from the left-hand panel to view parallel coordinates graphs and metrics comparison tables.
