import os
import mlflow
import numpy as np
import pandas as pd
import pytest


def test_model_prediction_schema_and_shape():
    """Verify that the production pipeline generates correctly bounded integer classification vectors."""
    # Resolve local SQLite data store configurations
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    db_path = os.path.join(PROJECT_ROOT, "mlflow.db")

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = mlflow.tracking.MlflowClient()

    # Pull active experiment structures
    experiment = client.get_experiment_by_name(
        "Provider_Reimbursement_Prediction"
    )
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    xgboost_run = next(
        r for r in runs if r.info.run_name == "XGBoost_Gradient_Boosting"
    )

    model_uri = (
        f"runs:/{xgboost_run.info.run_id}/production_reimbursement_pipeline"
    )
    model_pipeline = mlflow.pyfunc.load_model(model_uri)

    # Structure mock test frame inputs
    test_sample = pd.DataFrame(
        [
            {
                "provider_specialty": "Orthopedics",
                "facility_type": "Outpatient Clinic",
                "cpt_code": "27447",
                "state": "CA",
                "insurer_tier": "Platinum",
                "service_weight": 8.5,
                "submitted_charges": 8500.0,
                "target_budget": 3000.0,  # Deliberate low target budget
            }
        ]
    )

    prediction = model_pipeline.predict(test_sample)

    # Format validations
    assert len(prediction) == 1
    assert isinstance(prediction[0], (np.integer, int))
    assert prediction[0] in [
        0,
        1,
    ]  # Must conform to structural binary outputs matrix bounds


def test_model_performance_threshold():
    """Verify that the registered model correctly highlights extreme cost risk overruns."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    db_path = os.path.join(PROJECT_ROOT, "mlflow.db")

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(
        "Provider_Reimbursement_Prediction"
    )
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    xgboost_run = next(
        r for r in runs if r.info.run_name == "XGBoost_Gradient_Boosting"
    )

    model_uri = (
        f"runs:/{xgboost_run.info.run_id}/production_reimbursement_pipeline"
    )
    model_pipeline = mlflow.pyfunc.load_model(model_uri)

    # Highly expensive orthopedic procedure with an extremely tiny budget constraint
    high_risk_claim = pd.DataFrame(
        [
            {
                "provider_specialty": "Orthopedics",
                "facility_type": "Inpatient Hospital",
                "cpt_code": "27447",
                "state": "CA",
                "insurer_tier": "Platinum",
                "service_weight": 8.5,
                "submitted_charges": 11500.0,
                "target_budget": 500.0,  # Extreme target budget crunch
            }
        ]
    )

    prediction = model_pipeline.predict(high_risk_claim)

    # Performance threshold: Must flag this obvious overrun as a 1
    assert prediction[0] == 1
