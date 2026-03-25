# Fraud Detection API
# Triggering new deployment on Render
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection API")

# Setup CORS to allow the frontend to access the API
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the autoencoder model
try:
    autoencoder = load_model("model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    autoencoder = None

# Define the threshold from the notebook
THRESHOLD = 2.9

class TransactionData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running. Use POST /predict to test transactions."}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    if autoencoder is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Extract features in the correct order
    features = [
        data.V1, data.V2, data.V3, data.V4, data.V5, data.V6, data.V7,
        data.V8, data.V9, data.V10, data.V11, data.V12, data.V13, data.V14,
        data.V15, data.V16, data.V17, data.V18, data.V19, data.V20, data.V21,
        data.V22, data.V23, data.V24, data.V25, data.V26, data.V27, data.V28,
        data.Amount
    ]
    
    # Convert to numpy array and reshape to (1, 29)
    input_data = np.array(features).reshape(1, -1)
    
    # The notebook scales Amount using StandardScaler, but since we don't have the fitted scaler,
    # we'll use the raw Amount here. In a rigorous setup, we'd load a pickled scaler.
    
    # Predict reconstruction
    predictions = autoencoder.predict(input_data)
    
    # Calculate Mean Squared Error (reconstruction error)
    mse = np.mean(np.power(input_data - predictions, 2), axis=1)
    
    error = float(mse[0])
    is_fraud = error > THRESHOLD
    
    return {
        "is_fraud": is_fraud,
        "reconstruction_error": error,
        "threshold": THRESHOLD
    }
