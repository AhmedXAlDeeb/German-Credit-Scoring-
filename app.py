from fastapi import FastAPI, Request
import joblib
import pandas as pd
import shap
from pydantic import BaseModel

app = FastAPI()

# Load model and features
model = joblib.load("credit_model_rf.joblib")
feature_list = joblib.load("feature_list.joblib")
categorical_columns = joblib.load("categorical_columns.joblib")

# Load or define explainer
explainer = shap.Explainer(model)

# Final categorical features (before encoding)
final_categorical_features = [
    'checking_status_string', 'credit_history_string', 'purpose_string', 'savings_string', 'employment_string',
    'other_guarantors_string', 'property_string', 'other_installment_string', 'housing_string', 'job_string',
    'gender', 'marital_status', 'has_guarantor', 'is_unemployed', 'has_telephone', 'is_foreign', 'has_other_debts'
]

final_numerical_features = [
    'Duration', 'Credit amount', 'Installment rate in percentage of disposable income',
    'Present residence since', 'Age', 'Number of existing credits at this bank',
    'Number of people being liable to provide maintenance for'
]

# ------------------------- INPUT SCHEMA -------------------------
class CreditRequest(BaseModel):
    Duration: float
    Credit_amount: float
    Installment_rate_in_percentage_of_disposable_income: float
    Present_residence_since: float
    Age: float
    Number_of_existing_credits_at_this_bank: float
    Number_of_people_being_liable_to_provide_maintenance_for: float
    checking_status_string: str
    credit_history_string: str
    purpose_string: str
    savings_string: str
    employment_string: str
    other_guarantors_string: str
    property_string: str
    other_installment_string: str
    housing_string: str
    job_string: str
    gender: str
    marital_status: str
    has_guarantor: str
    is_unemployed: str
    has_telephone: str
    is_foreign: str
    has_other_debts: str

# ---------------------- PREPROCESS FUNCTION ---------------------
def preprocess_input(data_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])

    # Feature engineering
    df["CreditPerMonth"] = df["Credit_amount"] / df["Duration"]
    df["DebtLoad"] = df["Credit_amount"] / (df["Installment_rate_in_percentage_of_disposable_income"] + 1)

    # Separate and encode
    df_cat = pd.get_dummies(df[final_categorical_features], drop_first=True)
    df_cat = df_cat.reindex(columns=categorical_columns, fill_value=0)
    df_num = df[final_numerical_features + ["CreditPerMonth", "DebtLoad"]]

    # Combine
    df_combined = pd.concat([df_num, df_cat], axis=1)

    # Ensure correct column order
    df_combined = df_combined.reindex(columns=feature_list, fill_value=0)

    # Clean column names (if needed)
    df_combined.columns = (
        df_combined.columns
        .str.replace(r"[<>\[\]]", "", regex=True)
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    return df_combined

# ------------------------- ROUTES -------------------------

@app.post("/predict")
async def predict_creditability(data: CreditRequest):
    input_dict = data.dict()
    input_df = preprocess_input(input_dict)

    prediction = model.predict(input_df)[0]
    label = "Good" if prediction == 1 else "Bad"

    return {
        "prediction": label,
        "message": "The applicant is likely to be creditworthy." if prediction == 1 else "The applicant may pose a credit risk."
    }

@app.post("/predict_with_explanation")
async def predict_with_explanation(data: CreditRequest):
    input_dict = data.dict()
    input_df = preprocess_input(input_dict)

    pred_proba = model.predict_proba(input_df)[0][1]
    shap_values = explainer(input_df)

    explanation = shap_values.values[0]
    base_value = shap_values.base_values[0]
    contributions = dict(zip(input_df.columns, explanation))

    # Create readable summary
    top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    summary = "Prediction was influenced most by: " + ", ".join(
        [f"{f} (impact: {v:.3f})" for f, v in top_features]
    )

    return {
        "prediction_probability": round(pred_proba, 3),
        "base_value": round(base_value, 3),
        "top_feature_contributions": top_features,
        "explanation_summary": summary
    }

@app.get("/")
def get_info():
    return {
        "model_name": "Credit Scoring Random Forest",
        "version": "1.0.0",
        "description": "A Random Forest model for predicting creditworthiness, with SHAP explainability."
    }
