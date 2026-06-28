import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Resolve Project Root Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Configurations not found at {CONFIG_PATH}")
    sys.exit(1)

data_cfg = config["data_paths"]
split_cfg = config["split_parameters"]
models_cfg = config["models"]

# 2. Extract Data Paths (Points directly to raw dataset to fit transformers natively)
raw_data_path = os.path.join(PROJECT_ROOT, "data", "raw", "healthcare_raw.csv")

try:
    df_raw = pd.read_csv(raw_data_path)
except FileNotFoundError:
    print(f"Error: Missing raw tracking matrix at {raw_data_path}. Please run src/preprocess.py first.")
    sys.exit(1)

# Isolate training attributes from target labels
X = df_raw[[
    "provider_specialty", "facility_type", "cpt_code", 
    "state", "insurer_tier", "service_weight", 
    "submitted_charges", "target_budget"
]].copy()
# Enforce CPT column values match string data types safely
X["cpt_code"] = X["cpt_code"].astype(str)
y = df_raw[data_cfg["target_column"]]

# Split data strictly before setting transformers to preserve pipeline integrity
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split_cfg["test_size"], 
    random_state=split_cfg["random_state"], 
    stratify=y if split_cfg["stratify_target"] else None
)

# 3. Dynamic Preprocessing Pipeline Architecture
categorical_cols = ["provider_specialty", "facility_type", "cpt_code", "state", "insurer_tier"]
numerical_cols = ["service_weight", "submitted_charges", "target_budget"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# 4. Model Algorithm Blueprint Factory mapping
model_factory = {
    "logistic_regression": LogisticRegression,
    "logistic_regression_tuned": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "random_forest_deep": RandomForestClassifier,
    "gradient_boosting": XGBClassifier,
}

# Connect to the local SQL relational tracking database store
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
mlflow.set_experiment(config["mlflow_settings"]["experiment_name"])

print(f"--- Launching Unified MLflow Pipeline Experimentation ({len(models_cfg)} Models) ---\n")

# 5. Pipeline Execution Loop
for model_key, model_info in models_cfg.items():
    run_name = model_info["run_name"]
    hyperparameters = model_info["hyperparameters"]
    model_name = model_info["model_name"]

    print(f"Starting Training Session: {run_name}...")

    with mlflow.start_run(run_name=run_name):
        # Log training hyperparameters matrix parameters
        mlflow.log_params(hyperparameters)
        mlflow.log_param("algorithm_type", model_key)

        # Initialize core classifier model instance
        classifier_model = model_factory[model_key](**hyperparameters)

        # Build an end-to-end production pipeline packaging both preprocessing and modeling layers
        production_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier_model)
        ])

        # Fit both preprocessing scales and model boundaries together safely
        production_pipeline.fit(X_train, y_train)

        # Predict performance metrics using raw text data structures
        y_pred = production_pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Register the complete end-to-end inference pipeline artifact cleanly to database storage
        mlflow.sklearn.log_model(production_pipeline, name="production_reimbursement_pipeline")

        print(f"✔️ Finished {run_name}")
        print(f"   F1-Score: {metrics['f1_score']:.4f} | Recall: {metrics['recall']:.4f}\n")

print("All pipelines successfully registered! Run 'python src/app.py' to launch your web app interface.")
