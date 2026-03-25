# Fraud Detection API
# Triggering new deployment on Render
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import h5py
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Fraud Detection API")

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Load the autoencoder weights from model.h5 using h5py (no TF required)
# The model architecture from the notebook:
#   Input(29) -> Dense(32, relu) -> Dense(16, relu) -> Dense(8, relu)
#             -> Dense(16, relu) -> Dense(29, sigmoid)
# ---------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def load_weights(path):
    weights = {}
    with h5py.File(path, "r") as f:
        # Navigate to the model weights inside the h5 file
        # Common Keras H5 layout: model_weights -> layer_name -> layer_name -> kernel:0 / bias:0
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = obj[()]
        f.visititems(visit)
    return weights

def predict_autoencoder(weights, x):
    """
    Run a forward pass through the autoencoder using extracted weights.
    Layers: dense(relu), dense(relu), dense(relu), dense(relu), dense(sigmoid)
    """
    layer_order = []
    # Find all kernel weights and sort by layer name
    kernels = sorted([k for k in weights if 'kernel' in k])
    biases  = sorted([k for k in weights if 'bias'   in k])

    out = x
    activations = [relu, relu, relu, relu, sigmoid]
    for i, (k, b) in enumerate(zip(kernels, biases)):
        W = weights[k]
        B = weights[b]
        out = out @ W + B
        if i < len(activations):
            out = activations[i](out)
    return out

# Load weights at startup
MODEL_PATH = "model.h5"
THRESHOLD = 2.9
model_weights = None

try:
    model_weights = load_weights(MODEL_PATH)
    print(f"Model loaded successfully. Layers found: {len([k for k in model_weights if 'kernel' in k])}")
except Exception as e:
    print(f"Error loading model: {e}")


class TransactionData(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float


@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running. Use POST /predict to test transactions."}


@app.post("/predict")
def predict_fraud(data: TransactionData):
    if model_weights is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    features = [
        data.V1, data.V2, data.V3, data.V4, data.V5, data.V6, data.V7,
        data.V8, data.V9, data.V10, data.V11, data.V12, data.V13, data.V14,
        data.V15, data.V16, data.V17, data.V18, data.V19, data.V20, data.V21,
        data.V22, data.V23, data.V24, data.V25, data.V26, data.V27, data.V28,
        data.Amount
    ]

    x = np.array(features, dtype=np.float32).reshape(1, -1)
    reconstructed = predict_autoencoder(model_weights, x)
    mse = float(np.mean(np.power(x - reconstructed, 2)))
    is_fraud = mse > THRESHOLD

    return {
        "is_fraud": is_fraud,
        "reconstruction_error": mse,
        "threshold": THRESHOLD
    }
