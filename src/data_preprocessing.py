import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure directory structures exist for upcoming DVC tracking integration
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =====================================================================
# PIPELINE GENERATION BLOCK
# =====================================================================
np.random.seed(42)
num_records = 5000
num_providers = 150

provider_ids = [f"NPI_{100000 + i}" for i in range(num_providers)]
provider_specialties = [
    "Cardiology",
    "Orthopedics",
    "Family Medicine",
    "Pediatrics",
    "Dermatology",
]
provider_specialty_map = {
    pid: np.random.choice(provider_specialties, p=[0.15, 0.20, 0.35, 0.15, 0.15])
    for pid in provider_ids
}

states = ["NY", "CA", "TX", "FL", "IL"]
state_multipliers = {"NY": 1.25, "CA": 1.30, "TX": 0.90, "FL": 1.0, "IL": 1.05}
provider_state_map = {pid: np.random.choice(states) for pid in provider_ids}

chosen_providers = np.random.choice(provider_ids, size=num_records)
facility_types = ["Inpatient Hospital", "Outpatient Clinic", "Solo Practice"]
facilities = np.random.choice(facility_types, size=num_records, p=[0.3, 0.5, 0.2])

cpt_code_map = {
    "99213": 100,
    "99214": 150,
    "93000": 250,
    "27447": 5000,
    "99204": 200,
}
cpt_codes = np.random.choice(list(cpt_code_map.keys()), size=num_records)
insurer_tiers = ["Bronze", "Silver", "Gold", "Platinum"]
tiers = np.random.choice(insurer_tiers, size=num_records, p=[0.3, 0.4, 0.2, 0.1])

df_raw = pd.DataFrame(
    {
        "provider_id": chosen_providers,
        "facility_type": facilities,
        "cpt_code": cpt_codes,
        "insurer_tier": tiers,
    }
)
df_raw["provider_specialty"] = df_raw["provider_id"].map(provider_specialty_map)
df_raw["state"] = df_raw["provider_id"].map(provider_state_map)


def assign_service_weight(cpt):
    return 1.0 if cpt != "27447" else 8.5


df_raw["service_weight"] = df_raw["cpt_code"].apply(assign_service_weight)


def calculate_financials(row):
    base_fee = cpt_code_map[row["cpt_code"]]
    markup_factor = np.random.uniform(1.2, 2.5)
    submitted_charges = round(base_fee * row["service_weight"] * markup_factor, 2)

    spec_mod = 1.3 if row["provider_specialty"] in ["Cardiology", "Orthopedics"] else 1.0
    fac_mod = 1.4 if row["facility_type"] == "Inpatient Hospital" else 1.0
    geo_mod = state_multipliers[row["state"]]
    tier_map = {"Bronze": 0.85, "Silver": 1.0, "Gold": 1.15, "Platinum": 1.3}
    tier_mod = tier_map[row["insurer_tier"]]

    allowed_amount = round(
        base_fee * row["service_weight"] * spec_mod * fac_mod * geo_mod * tier_mod,
        2,
    )

    lesser_of_cap = min(submitted_charges, allowed_amount)
    policy_ceiling = 12000.00
    final_reimbursement = min(lesser_of_cap, policy_ceiling)

    is_capped_by_charges = 1 if submitted_charges < allowed_amount else 0
    is_capped_by_policy = 1 if lesser_of_cap > policy_ceiling else 0
    percent_of_charge = round((final_reimbursement / submitted_charges) * 100, 2)

    return pd.Series(
        [
            submitted_charges,
            allowed_amount,
            percent_of_charge,
            final_reimbursement,
            is_capped_by_charges,
            is_capped_by_policy,
        ]
    )


df_raw[
    [
        "submitted_charges",
        "allowed_amount",
        "percent_of_charge",
        "final_reimbursement_rate",
        "is_capped_by_charges",
        "is_capped_by_policy",
    ]
] = df_raw.apply(calculate_financials, axis=1)
df_raw["target_budget"] = df_raw["final_reimbursement_rate"].apply(
    lambda x: round(x * np.random.uniform(0.85, 1.15), 2)
)
df_raw["exceeds_budget"] = (
    df_raw["final_reimbursement_rate"] > df_raw["target_budget"]
).astype(int)

# Inject null artifacts representing common human administrative error gaps
df_raw.loc[df_raw.sample(frac=0.03).index, "facility_type"] = np.nan
df_raw.loc[df_raw.sample(frac=0.03).index, "submitted_charges"] = np.nan

# EXPORT RAW UNALTERED ARTIFACT
df_raw.to_csv("data/raw/healthcare_raw.csv", index=False)

# =====================================================================
# PREPROCESSING BLOCK
# =====================================================================
categorical_cols = [
    "provider_specialty",
    "facility_type",
    "cpt_code",
    "state",
    "insurer_tier",
]
numerical_cols = ["service_weight", "submitted_charges", "target_budget"]
target_col = ["exceeds_budget"]

df_cleaned = df_raw.copy()
df_cleaned[numerical_cols] = SimpleImputer(strategy="median").fit_transform(
    df_cleaned[numerical_cols]
)
df_cleaned[categorical_cols] = SimpleImputer(
    strategy="most_frequent"
).fit_transform(df_cleaned[categorical_cols])

scaled_numerical = StandardScaler().fit_transform(df_cleaned[numerical_cols])
df_numerical_scaled = pd.DataFrame(scaled_numerical, columns=numerical_cols)

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_categorical = encoder.fit_transform(df_cleaned[categorical_cols])
df_categorical_encoded = pd.DataFrame(
    encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols)
)

# Reassemble Clean Training Matrix
df_preprocessed_master = pd.concat(
    [
        df_numerical_scaled,
        df_categorical_encoded,
        df_cleaned[target_col].reset_index(drop=True),
    ],
    axis=1,
)

# EXPORT PREPROCESSED MATRIX
df_preprocessed_master.to_csv(
    "data/processed/healthcare_preprocessed_master.csv", index=False
)

print("Data exported successfully into standard directories!")
print("-> Raw Output Path: data/raw/healthcare_raw.csv")
print("-> Processed Output Path: data/processed/healthcare_preprocessed_master.csv")
