
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from prediction.schema import CustomerData


# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Connect to MLflow server running on Windows host
model_path = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_pusher",
    "model.pkl"
)

model = joblib.load(model_path)


# Load fitted preprocessor
preprocessor_path = os.path.join(
    BASE_DIR,
    "artifacts",
    "model_pusher",
    "preprocessor.pkl"
)

preprocessor = joblib.load(preprocessor_path)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:
        # Convert incoming customer data into DataFrame
        input_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Apply fitted preprocessing
        transformed_data = preprocessor.transform(input_data)

        # Generate prediction
        prediction = model.predict(transformed_data)

        result = int(prediction[0])

        return {
            "prediction": result,
            "churn_status": (
                "Churn" if result == 1 else "No Churn"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
