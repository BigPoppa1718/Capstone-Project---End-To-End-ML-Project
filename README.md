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

## Evaluation

## 🏆 Model Selection & Production Justification Analysis

### 1. Metric Performance Summary on Held-Out Test Set
The model configurations were evaluated using a strict 20% holdout validation partition (1,000 completely unseen records) stratified to prevent class imbalance skew. 

*   **XGBoost Gradient Boosting**: Achieved an **Accuracy of 0.7520**, **Precision of 0.7258**, **Recall of 0.7583**, **F1-Score of 0.7416**, and an **ROC-AUC of 0.8145**.
*   **Ensemble Random Forest**: Achieved an **Accuracy of 0.7380**, **Precision of 0.7311**, **Recall of 0.7210**, **F1-Score of 0.7260**, and an **ROC-AUC of 0.7932**.
*   **Baseline Logistic Regression**: Achieved an **Accuracy of 0.7090**, **Precision of 0.6765**, **Recall of 0.7642**, **F1-Score of 0.7177**, and an **ROC-AUC of 0.7512**.

### 2. Final Selection & Core Engineering Justification
**XGBoost Gradient Boosting is selected as the winning production engine** for Component 2 of this Capstone project. The engineering justification rests on two primary operational pillars:

1.  **Non-Linear Constraint Optimization**: The healthcare pricing pipeline contains strict mathematical cliffs introduced during preprocessing (e.g., the contractual maximum `$12,000` ceiling and the "Lesser Of" billing rule). While Logistic Regression struggled to establish clean linear decision lines, XGBoost's iterative residual boosting algorithm naturally mapped these structural inflection points, resulting in a superior classification capacity (**ROC-AUC: 0.8145**).
2.  **Harmonic Performance Stability**: While Logistic Regression yielded a marginally higher raw Recall metric (`0.7642` vs. XGBoost's `0.7583`), it suffered from severe precision degradation. Selecting Logistic Regression would result in an excessive volume of False Alarms—conversely over-predicting that providers will blow past their target budgets when they are actually safe. XGBoost optimizes the trade-off, providing the peak **F1-Score (0.7416)** necessary to run stable automated financial auditing inside the conversational LLM layer.


### LLM- Powered Interface
# Phase 3: Conversational LLM-Powered Interface Layer (Component 2)

This phase implements the natural language interface layer (Component 2) of the Capstone project. It wraps our highest-performing registered machine learning model inside a responsive **Streamlit Web Application**, orchestrating user query extraction, edge-case validation, model inference execution, and dynamic response generation using **Nebius AI Studio**.

---

## 🏗️ Interface Architecture & Orchestration

The application uses an advanced **Function-Calling & Structured Outputs Architecture** to create a predictable bridge between raw user text and the machine learning model.

```text
[User Plain English Text]
         │
         ▼
 ┌───────────────┐
 │   Nebius LLM  │ ──► Parses structured parameters using Pydantic schema validation
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │ XGBoost Model │ ──► Runs inference on extracted parameters against target budget
 └───────────────┘
         │
         ▼
 ┌───────────────┐
 │   Nebius LLM  │ ──► Translates binary prediction into context-aware risk analysis
 └───────────────┘
         │
         ▼
[Conversational UI Response]
```

### 1. Robust Input Parsing (Pydantic Layer)
To prevent model failures from bad data entry, the interface uses a strict structural extraction layer backed by `pydantic` and Nebius AI (`meta-llama/Meta-Llama-3.1-70B-Instruct`). The LLM acts as a dynamic entity extractor, converting unstructured sentences into a structured JSON payload that matches our data pipeline training schema.

### 2. Guardrails & Edge Case Handling
The interface includes business logic designed to intercept bad inputs before they hit the model pipeline:
* **Scope Validation (`is_valid_query`)**: If a user submits text entirely unrelated to medical claims (e.g., asking for a recipe or travel advice), the system identifies the violation, alerts the user, and skips model execution.
* **Completeness Validation (`missing_information`)**: If critical billing variables are missing (e.g., omitting the `state`, `target_budget`, or `submitted_charges`), the system provides a detailed description of what data is needed to clear ambiguity instead of passing garbage data to the model.

### 3. Dynamic Context Synthesis
Once the underlying XGBoost pipeline finishes its classification pass, the prediction indicator (`1` for Overrun, `0` for Under Budget) is passed back to Nebius AI along with the original text parameters. The LLM translates the raw binary output into a professional advisory brief, contextualizing the impact of regional multipliers and plan tiers while noting contract caps (like the `$12,000` rule).

---

## 🛡️ Production Security: Environment Secrets

To comply with enterprise security architectures, the system strictly separates access tokens from source code repository files:
* **No Hardcoding**: The application does not contain hardcoded API tokens. 
* **Environment Binding**: The script looks for the native environment variable `NEBIUS_API_KEY`. If it is missing, the interface shuts down gracefully with a clean initialization error.

---

## 📦 Containerization & Execution Guide

### 1. Running Locally (Standard Shell)
Ensure you have your environment key set up inside your active terminal workspace before launching the application:

```bash
# Bind your unique Nebius AI access token
export NEBIUS_API_KEY="your_actual_nebius_api_key_here"

# Initialize the Streamlit engine
streamlit run app.py
```

### 2. Production Docker Deployment
Containerizing the application abstracts all runtime compilation, including installing C++ multithreading wrappers (`libomp`) required for XGBoost inside Linux environments.

```bash
# Build the unified production image container
docker build -t provider-billing-assistant .

# Run the container while forwarding web traffic ports and passing secrets
docker run -p 8501:8501 -e NEBIUS_API_KEY=\$NEBIUS_API_KEY provider-billing-assistant
```
Once initialized, navigate to `http://localhost:8501` inside your web browser to interact with your end-to-end intelligent app interface.


** cloud runner environments upon every code modification.

---

## 🛠️ Core Testing Matrices

The verification matrix spans three isolated domains covering data transformations, artifact state conditions, and application boundary schemas:

### 1. Preprocessing Verification Layer (`tests/test_preprocess.py`)
Validates that our numerical and categorical transformers handle common Electronic Health Record (EHR) entry errors correctly before feed-forwarding matrices to the model:
* **Missing Value Imputation**: Asserts that `SimpleImputer` successfully captures artificially injected null columns (`NaN`) and maps placeholder rows without losing feature data indices.
* **Categorical Feature Expansion**: Checks that string parameters (specialties, facility spaces, insurer levels) convert cleanly into expanded one-hot binary dummy variables matching expected matrix dimensions.
* **Structure Immutability Protection**: Deep-snapshots original incoming data structures to prove that our cleaning functions do not cause any global mutations or predictive data leakage.

### 2. Model Performance & Output Testing (`tests/test_model.py`)
Validates that models serialized via MLflow and the local relational SQLite store (`mlflow.db`) fulfill production runtime conditions:
* **Output Type/Shape Validation**: Inspects incoming predictions to guarantee that our winning XGBoost algorithm emits bounded binary classification types (`[0, 1]`) instead of continuous floating decimals.
* **Operational Performance Threshold Assertions**: Submits an extreme, obvious target budget deficit case to confirm the classifier flags high-risk financial overruns with 100% sensitivity.

### 3. LLM Parsing & Schema Testing (`tests/test_interface.py`)
Guarantees that input schemas managed by Pydantic and Nebius AI Studio parse fields safely without breaking backend calculators:
* **Structured Variable Parsing Accuracy**: Confirms that clean input blocks map perfectly to predefined Pydantic object variables.
* **Adversarial Input/Type Handling**: Simulates unparsable string entry blocks inside numerical fields to verify that the Pydantic validator throws an explicit `ValidationError` to block broken queries.

---

## 🚀 Execution Environment Instructions

### Local Test Automation Execution
To manually run the validation framework on your development machine, add your project workspace path to Python's internal search path:

```bash
python -m pytest tests/ -v
```

### Continuous Integration (CI/CD GitHub Actions)
Every code contribution pushed to the remote repository activates the automated workflow runner (`.github/workflows/mlops.yaml`). The cloud environment spins up an isolated Ubuntu machine, sets up Python 3.12, installs system multithreading dependencies (`libomp`), builds packages from `requirements.txt`, processes data states, and executes the complete `pytest` validation loop automatically.
