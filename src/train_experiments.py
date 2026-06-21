import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Dynamic Path Fix: Locates config.yaml relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Configurations not found at {CONFIG_PATH}")
    sys.exit(1)

# Dynamic Path Fix: Ensures data input path resolves correctly relative to the project root
data_cfg = config["data_paths"]
processed_data_path = os.path.join(PROJECT_ROOT, data_cfg["processed_data_input"])

# Extract pipeline elements
split_cfg = config["split_parameters"]
models_cfg = config["models"]

# Add this line to enforce an SQLite database backend store
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Initialize unified MLflow setup
mlflow.set_experiment(config["mlflow_settings"]["experiment_name"])

# 2. Data Preparation
try:
    df = pd.read_csv(processed_data_path)
except FileNotFoundError:
    print(f"Error: Missing tracking matrix at {processed_data_path}")
    sys.exit(1)

X = df.drop(columns=[data_cfg["target_column"]])
y = df[data_cfg["target_column"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=split_cfg["test_size"],
    random_state=split_cfg["random_state"],
    stratify=y if split_cfg["stratify_target"] else None,
)

# 3. Algorithm Mapping Dictionary
model_factory = {
    "logistic_regression": LogisticRegression,
    "logistic_regression_tuned": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "random_forest_deep": RandomForestClassifier,
    "gradient_boosting": XGBClassifier,
}

print(f"--- Launching MLflow Experimentation Matrix ({len(models_cfg)} Models) ---\n")

# 4. Multi-Model Training Loop
for model_key, model_info in models_cfg.items():
    run_name = model_info["run_name"]
    hyperparameters = model_info["hyperparameters"]
    model_name = model_info["model_name"]

    print(f"Starting Training Session: {run_name}...")

    with mlflow.start_run(run_name=run_name):
        # A. Track hyperparameters
        mlflow.log_params(hyperparameters)
        mlflow.log_param("algorithm_type", model_key)

        # B. Instantiate model from dictionary mapping
        model_class = model_factory[model_key]
        model_instance = model_class(**hyperparameters)

        # C. Execute fitting
        model_instance.fit(X_train, y_train)

        # D. Predict and Evaluate
        y_pred = model_instance.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),  # Critical budget metric
            "f1_score": f1_score(y_test, y_pred),
        }

        # E. Log Metrics to local MLflow database
        mlflow.log_metrics(metrics)

        # F. Log Model Binaries safely using the modernized API parameters
        if model_key == "gradient_boosting":
            mlflow.xgboost.log_model(model_instance, name=model_name)
        else:
            mlflow.sklearn.log_model(model_instance, name=model_name)

        print(f"✔️ Finished {run_name}")
        print(f"   F1-Score: {metrics['f1_score']:.4f} | Recall: {metrics['recall']:.4f}\n")

print("All evaluations registered! Open 'mlflow ui' to audit the comparison table.")
