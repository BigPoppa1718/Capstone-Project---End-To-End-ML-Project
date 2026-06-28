import os
import json
import yaml
import mlflow
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# ==========================================
# 1. INITIALIZATION & SECURITY ENVIRONMENT
# ==========================================
st.set_page_config(page_title="Healthcare Reimbursement Assistant", page_icon="🩺", layout="centered")

# Ensure API Key is bound securely from environment vectors
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

# FIX: Set the correct official Nebius Studio API connection endpoint
BASE_URL = "https://api.studio.nebius.ai/v1"

@st.cache_resource
def load_production_artifacts():
    """Loads shared project configurations and the winning registered ML pipeline."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    db_path = os.path.join(PROJECT_ROOT, "mlflow.db")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = mlflow.tracking.MlflowClient()
    
    experiment = client.get_experiment_by_name(config["mlflow_settings"]["experiment_name"])
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Locate the optimal run (XGBoost)
    xgboost_run = next(r for r in runs if r.info.run_name == "XGBoost_Gradient_Boosting")
    
    # FIX: Load the unified pipeline artifact containing your preprocessor + model transformer layers
    model_uri = f"runs:/{xgboost_run.info.run_id}/production_reimbursement_pipeline"
    
    loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
    return config, loaded_pipeline


try:
    config, model = load_production_artifacts()
except Exception as e:
    st.error(f"⚠️ Failed to load production ML models. Ensure 'mlflow.db' is populated. Error: {e}")
    st.stop()

# ==========================================
# 2. DEFINING STRUCTURED PARSING SCHEMA
# ==========================================
class ClaimFeatures(BaseModel):
    is_valid_query: bool = Field(description="False if the query is unrelated to healthcare, billing, or completely empty.")
    missing_information: Optional[str] = Field(description="List missing features if user input is incomplete (e.g. state, charges). Null if complete.")
    provider_specialty: Optional[str] = Field(description="Must be exactly: Cardiology, Orthopedics, Family Medicine, Pediatrics, or Dermatology.")
    facility_type: Optional[str] = Field(description="Must be exactly: Inpatient Hospital, Outpatient Clinic, or Solo Practice.")
    cpt_code: Optional[str] = Field(description="Must be string format of code: 99213, 99214, 93000, 27447, or 99204.")
    state: Optional[str] = Field(description="Two-letter upper state code: NY, CA, TX, FL, IL.")
    insurer_tier: Optional[str] = Field(description="Must be exactly: Bronze, Silver, Gold, or Platinum.")
    submitted_charges: Optional[float] = Field(description="Raw numerical dollar charges billed by provider.")
    target_budget: Optional[float] = Field(description="The financial budget cap threshold to evaluate against.")

# ==========================================
# 3. STREAMLIT USER INTERFACE LAYOUT
# ==========================================
st.title("🩺 Provider Reimbursement Risk Assistant")
st.markdown("Enter standard healthcare claim details in plain English to evaluate budget overrun risks.")

user_query = st.text_area(
    "Describe your claim request:",
    placeholder="Example: I have an Orthopedics case at an Outpatient Clinic in CA for CPT code 27447. Billed charges are $8500 and my target budget is $7000. Insurer tier is Platinum.",
    height=120
)

if st.button("Analyze Financial Risk", type="primary"):
    if not NEBIUS_API_KEY:
        st.error("🔒 Security Key Error: `NEBIUS_API_KEY` environment variable is missing.")
        st.stop()
        
    if not user_query.strip():
        st.warning("Please type a valid plain English claim profile descriptor query.")
        st.stop()

    # Initialize OpenAI-compatible Nebius Client
    ai_client = OpenAI(base_url=BASE_URL, api_key=NEBIUS_API_KEY)
    
    with st.spinner("Parsing medical parameters and evaluating financial risk variables..."):
        # A. Structured input parsing layer via LLM function call routing
        system_prompt = (
            "You are an expert healthcare financial data engineer. Your job is to extract raw data features from text "
            "and format them strictly into the requested structural JSON format matching the ClaimFeatures schema."
        )
        
         # 1. FIX: Update the model string for the Structured Output Parse Layer (Around Line 89)
        try:
            completion = ai_client.beta.chat.completions.parse(
                model="meta-llama/Llama-3.3-70B-Instruct", # Updated to active Nebius endpoint identifier
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                response_format=ClaimFeatures,
                temperature=0.0
            )
            parsed_data = completion.choices[0].message.parsed
        except Exception as err:
            st.error(f"Failed to communicate with extraction LLM: {err}")
            st.stop()

        # B. Edge Case Handling Logic
        if not parsed_data.is_valid_query:
            st.error("❌ Out of Scope: Your query does not appear to relate to healthcare billing or reimbursement claims auditing.")
            st.stop()
            
        if parsed_data.missing_information:
            st.warning(f"⚠️ **Incomplete Input Variables Detected:** {parsed_data.missing_information}")
            st.info("Please expand your text block to supply the required billing elements to clear calculation ambiguities.")
            st.stop()

        # ==========================================
        # C. ML Model Pipeline Invocation Engine
        # ==========================================
        # Hardened Guardrail: Intercept missing numeric fields before casting to float
        if parsed_data.submitted_charges is None or parsed_data.target_budget is None:
            st.warning("⚠️ **Incomplete Numerical Data:** The system could not extract explicit values for submitted charges or target budgets.")
            st.info("Please expand your query text to provide clear dollar amounts (e.g., 'Charges are $450 and budget is $300') to run the financial risk calculator.")
            st.stop()

        # Structure fields safely now that numeric values are verified present
        input_payload = pd.DataFrame([{
            "provider_specialty": parsed_data.provider_specialty,
            "facility_type": parsed_data.facility_type,
            "cpt_code": str(parsed_data.cpt_code) if parsed_data.cpt_code else "99213",
            "state": parsed_data.state if parsed_data.state else "NY",
            "insurer_tier": parsed_data.insurer_tier if parsed_data.insurer_tier else "Silver",
            "service_weight": 8.5 if str(parsed_data.cpt_code) == "27447" else 1.0,
            "submitted_charges": float(parsed_data.submitted_charges),
            "target_budget": float(parsed_data.target_budget)
        }])
        
        # Execute binary classification prediction seamlessly through the pipeline wrapper
        raw_prediction = int(model.predict(input_payload)[0])

        
        # D. Response Generation Layer
        response_prompt = f"""
        You are a senior healthcare financial consultant. Review the following details and write a clear context analysis response.
        
        - User Original Query: "{user_query}"
        - Extracted Parameters: {input_payload.to_json(orient='records')}
        - Machine Learning Risk Prediction Code: {raw_prediction} (Note: 1 means the reimbursement WILL EXCEED the target budget, 0 means it MEETS or stays under budget)
        
        Requirements for your message:
        1. Explicitly state the result in bold at the start (Under Budget vs Budget Overrun Risk).
        2. Explain the real-world financial context of why this happened based on the variables (e.g. specialty multipliers, geographic variations).
        3. Outline relevant data limitations (e.g. Model evaluates a 5,000 baseline rows synthetic contract, subject to dynamic local payer negotiations, and assumes a hard $12,000 macro contract ceiling).
        """
        
        #Update the model string for the Final Response Context Layer (Around Line 129)
        try:
            final_response = ai_client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct", # Updated to active Nebius endpoint identifier
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.3
            )
            st.chat_message("assistant", avatar="🩺").write(final_response.choices[0].message.content)
        except Exception as err:
            st.error(f"Failed to generate narrative summary context: {err}")
