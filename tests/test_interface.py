import pytest
from pydantic import ValidationError
from src.app import ClaimFeatures


def test_interface_extracts_correct_feature_values():
    """Verify that valid schema structures populate Pydantic model configurations successfully."""
    valid_payload = {
        "is_valid_query": True,
        "missing_information": None,
        "provider_specialty": "Cardiology",
        "facility_type": "Outpatient Clinic",
        "cpt_code": "93000",
        "state": "NY",
        "insurer_tier": "Gold",
        "submitted_charges": 250.00,
        "target_budget": 300.00,
    }

    # Instantiation check
    parsed_claim = ClaimFeatures(**valid_payload)
    assert parsed_claim.is_valid_query is True
    assert parsed_claim.provider_specialty == "Cardiology"
    assert parsed_claim.submitted_charges == 250.00


def test_interface_handles_invalid_inputs_gracefully():
    """Verify that type enforcement flags garbage numerical structures instantly."""
    invalid_payload = {
        "is_valid_query": True,
        "missing_information": "Missing charges field configuration mapping",
        "provider_specialty": "Cardiology",
        "facility_type": "Outpatient Clinic",
        "cpt_code": "93000",
        "state": "NY",
        "insurer_tier": "Gold",
        "submitted_charges": "This is non-numeric garbage text entry",  # Type failure point
        "target_budget": 300.00,
    }

    # The interface parsing schema should catch this formatting mismatch and throw a ValidationError
    with pytest.raises(ValidationError):
        ClaimFeatures(**invalid_payload)
