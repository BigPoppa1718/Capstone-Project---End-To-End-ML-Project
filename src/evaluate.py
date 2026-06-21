import json
import os
import sys
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# 1. Resolve Project Root Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

try:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: Configurations not found at {CONFIG_PATH}")
    sys.exit(1)

data_cfg = config["data_paths"]
split_cfg = config["split_parameters"]
metrics_cfg = config["metrics_logging"]
processed_data_path = os.path.join(PROJECT_ROOT, data_cfg["processed_data_input"])

# 2. Re-create the Identical Held-Out Partition
try:
    df = pd.read_csv(processed_data_path)
except FileNotFoundError:
    print(f"Error: Missing processed data at {processed_data_path}")
    sys.exit(1)

X = df.drop(columns=[data_cfg["target_column"]])
y = df[data_cfg["target_column"]]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=split_cfg["test_size"],
    random_state=split_cfg["random_state"],
    stratify=y if split_cfg["stratify_target"] else None,
)

# 3. Connect to the Relational MLflow Store
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name(config["mlflow_settings"]["experiment_name"])
if not experiment:
    print("Error: No tracked experiments discovered inside database.")
    sys.exit(1)

runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# 4. Master Evaluation Summary Matrix List
print("=====================================================================")
print("📊 EVALUATING ALL ACTIVE LOGGED EXPERIMENT CONFIGURATIONS")
print("=====================================================================")

all_model_metrics = {}
best_f1 = -1.0
best_model_report = {}

for run in runs:
    run_name = run.info.run_name
    run_id = run.info.run_id
    
    # Extract the algorithm type parameter logged during the loop iteration
    algo_type = run.data.params.get("algorithm_type", "unknown")
    
    # Retrieve the clean model name specified by our config profile
    model_name_key = config["models"].get(algo_type, {}).get("model_name", "model")
    model_uri = f"runs:/{run_id}/{model_name_key}"
    
    try:
        # Load binary using generic wrapper interface
        model = mlflow.pyfunc.load_model(model_uri)
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = y_pred
        
        # Calculate scores
        report_dict = classification_report(y_test.values, y_pred, output_dict=True)
        auc_score = float(roc_auc_score(y_test.values, y_proba))
        
        current_f1 = report_dict["macro avg"]["f1-score"]
        all_model_metrics[run_name] = {
            "accuracy": report_dict["accuracy"],
            "macro_f1": current_f1,
            "macro_recall": report_dict["macro avg"]["recall"],
            "roc_auc": auc_score
        }
        
        print(f"✔️ Successfully evaluated: {run_name} (F1: {current_f1:.4f})")
        
        # Keep track of the absolute winner for your structural JSON export file
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_report = report_dict
            best_model_report["roc_auc_score"] = auc_score
            best_model_report["winning_run_name"] = run_name

    except Exception as e:
        # Skip runs containing experimental variations or empty runs
        continue

# 5. Export Top Winning Run Metrics to File Location
metrics_dir = os.path.join(PROJECT_ROOT, metrics_cfg["metrics_dir"])
os.makedirs(metrics_dir, exist_ok=True)
metrics_file_path = os.path.join(metrics_dir, metrics_cfg["metrics_name"])

with open(metrics_file_path, "w") as json_file:
    json.dump(best_model_report, json_file, indent=4)

print("=====================================================================")
print(f"🎉 Production baseline evaluation logged to: {metrics_file_path}")
print("=====================================================================")
