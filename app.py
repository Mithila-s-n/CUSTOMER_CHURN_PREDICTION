# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import pandas as pd
import joblib

print("Step 1: Imports completed")

# Create FastAPI App
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn",
    version="1.0"
)

print("Step 2: Loading model...")

# Load Model
try:
    model = joblib.load("churn_model.joblib")
    print("✅ Step 3: Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

print("Step 4: Creating Request Model...")

# Input Schema
class CustomerData(BaseModel):
    Age: int
    Gender: int
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int
    Contract_Length: int
    Total_Spend: float
    Last_Interaction: int

print("Step 5: API Ready")

# Home Endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running!"
    }

# Prediction Endpoint
@app.post("/predict")
def predict(data: CustomerData):

    if model is None:
        return {
            "error": "Model not loaded"
        }

    input_data = pd.DataFrame([{
        "Age": data.Age,
        "Gender": data.Gender,
        "Tenure": data.Tenure,
        "Usage Frequency": data.Usage_Frequency,
        "Support Calls": data.Support_Calls,
        "Payment Delay": data.Payment_Delay,
        "Subscription Type": data.Subscription_Type,
        "Contract Length": data.Contract_Length,
        "Total Spend": data.Total_Spend,
        "Last Interaction": data.Last_Interaction
    }])

    prediction = model.predict(input_data)

    result = "Customer Will Churn" if prediction[0] == 1 else "Customer Will Not Churn"

    return {
        "prediction": int(prediction[0]),
        "result": result
    }