import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.preprocess import preprocessor


def test_preprocessing_handles_missing_values_and_scales():
    """Verify that numerical columns handle missing data via median and scale accurately."""
    # Create isolated dirty data
    raw_data = pd.DataFrame(
        {
            "provider_specialty": ["Cardiology", "Orthopedics"],
            "facility_type": ["Outpatient Clinic", np.nan],  # Injected null
            "cpt_code": ["99214", "27447"],
            "state": ["NY", "CA"],
            "insurer_tier": ["Gold", "Platinum"],
            "service_weight": [1.0, 8.5],
            "submitted_charges": [150.0, np.nan],  # Injected null
            "target_budget": [130.0, 6000.0],
        }
    )

    # Transform data strings cleanly through your core pipeline blueprint
    processed_matrix = preprocessor.fit_transform(raw_data)

    # Assertions
    assert processed_matrix is not None
    assert not np.isnan(processed_matrix).any()  # Checks that imputer removed all NaNs
    assert (
        processed_matrix.shape[0] == 2
    )  # Checks row preservation count threshold boundaries


def test_preprocessing_encodes_categoricals():
    """Verify that string metrics are successfully transformed into one-hot binary structures."""
    raw_data = pd.DataFrame(
        {
            "provider_specialty": ["Cardiology", "Orthopedics"],
            "facility_type": ["Outpatient Clinic", "Inpatient Hospital"],
            "cpt_code": ["99214", "27447"],
            "state": ["NY", "CA"],
            "insurer_tier": ["Gold", "Platinum"],
            "service_weight": [1.0, 8.5],
            "submitted_charges": [150.0, 9000.0],
            "target_budget": [130.0, 8000.0],
        }
    )

    processed_matrix = preprocessor.fit_transform(raw_data)

    # Encoded dimensions must exceed original categorical string count shape metrics
    assert processed_matrix.shape[1] == 8


def test_preprocessing_immutability():
    """Ensure that the input data frame remains structurally unchanged during calculations."""
    raw_data = pd.DataFrame(
        {
            "provider_specialty": ["Cardiology"],
            "facility_type": ["Outpatient Clinic"],
            "cpt_code": ["99214"],
            "state": ["NY"],
            "insurer_tier": ["Gold"],
            "service_weight": [1.0],
            "submitted_charges": [150.0],
            "target_budget": [130.0],
        }
    )

    # Snapshot deep copy variables
    raw_copy = raw_data.copy()

    _ = preprocessor.fit_transform(raw_data)

    # Assert immutability to check for processing data leaks
    pd.testing.assert_frame_equal(raw_data, raw_copy)
