from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Pydantic model for input
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Load model
model = joblib.load("model/model_v1.pkl")

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])
    return df

@app.post("/predict")
def predict(data: CustomerData):
    df = preprocess_input(data.dict())
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}


@app.get("/")
def root():
    return {"status": "AutoOps API is running 🚀"}

# Optional: to run locally via script
if __name__ == "__main__":
    uvicorn.run("deploy.app:app", host="0.0.0.0", port=8000, reload=True)

