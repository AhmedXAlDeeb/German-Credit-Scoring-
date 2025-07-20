
# 🧠 German Credit Scoring API

A FastAPI-based REST API that serves a trained Random Forest model for German credit scoring.  
It predicts whether a credit applicant is **Good** or **Bad** based on input financial and personal details.

---

## 🚀 How to Run the API Using Docker

### ✅ 1. Clone or Download the Repository

```bash
git clone <your-repo-url>
cd German-Credit-Scoring
````

### ✅ 2. Build the Docker Image

```bash
docker build -t credit-model-api .
```

### ✅ 3. Run the Docker Container

```bash
docker run -p 8000:8000 credit-model-api
```

Once running, access the API at:

* **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check (optional)**: [http://localhost:8000/](http://localhost:8000/)

---

## 📬 Example Request

You can try the model via Swagger UI or using `curl` or Postman.

### 🔁 Endpoint

```
POST /predict
```

### 📥 JSON Input Example

```json
{
  "Duration": 12,
  "Credit amount": 1000,
  "Age": 35,
  "checking_status_string": "no checking",
  "credit_history_string": "critical/other existing credit",
  "purpose_string": "radio/tv",
  "savings_string": "unknown/ no savings account",
  "employment_string": "1<=X<4",
  "other_guarantors_string": "none",
  "property_string": "car",
  "other_installment_string": "none",
  "housing_string": "own",
  "job_string": "skilled",
  "gender": "male",
  "marital_status": "single",
  "has_guarantor": false,
  "is_unemployed": false,
  "has_telephone": true,
  "is_foreign": false,
  "has_other_debts": false
}
```

### 📤 Response

```json
{
  "prediction": "Good"
}
```

---

## 📁 Project Structure

```
.
├── app.py                  # FastAPI application
├── Dockerfile              # Docker build instructions
├── requirements.txt        # Python dependencies
├── credit_model_rf.joblib  # Trained RandomForest model
├── feature_list.joblib     # List of features used in training
├── README.md               # Project documentation
```

---

## 🛠 Requirements

* Docker Desktop (WSL2 or Hyper-V)
* Python (if testing locally without Docker)

---

## ✨ Future Improvements

* Add input validation using Pydantic
* Log requests and predictions
* Deploy on Render / Hugging Face / Railway

---

## 🤝 License

MIT License. Feel free to modify and use!

```

