# app.py
from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("credit_model_rf.joblib")
feature_list = joblib.load("feature_list.joblib")  # save it during training

@app.post("/predict")
async def predict_creditability(data: dict):
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=feature_list, fill_value=0)

    prediction = model.predict(input_df)[0]
    prediction_label = "Good" if prediction == 1 else "Bad"

    return {"prediction": prediction_label}
