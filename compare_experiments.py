import mlflow
import pandas as pd

# 1. Connect to local backend storage
mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "Provider_Reimbursement_Prediction"
PRIMARY_METRIC = "metrics.f1_score"

# 2. Extract experiment details
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    print(f"Error: Experiment '{EXPERIMENT_NAME}' not found.")
    exit(1)

# 3. Programmatically search all runs using the MLflow API
df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if df_runs.empty:
    print("No runs found in this experiment context.")
    exit(1)

# 4. Filter, sort, and identify the top performer
df_sorted = df_runs.sort_values(by=PRIMARY_METRIC, ascending=False)
best_run = df_sorted.iloc[0]

print("=====================================================================")
print("🏆 PROGRAMMATIC EXPERIMENT COMPARISON SUMMARY")
print("=====================================================================")
print(f"Total Runs Analyzed:      {len(df_runs)}")
print(f"Best Performing Run:     {best_run['tags.mlflow.runName']}")
print(f"Winning Algorithm Type:   {best_run['params.algorithm_type']}")
print(f"Peak F1-Score Achieved:  {best_run[PRIMARY_METRIC]:.4f}")
print(f"Associated Recall Value:  {best_run['metrics.recall']:.4f}")
print(f"MLflow Run ID Pointer:    {best_run['run_id']}")
print("=====================================================================")
