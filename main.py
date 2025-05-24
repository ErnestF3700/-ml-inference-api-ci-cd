# main.py
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent / "model" / "iris_rf.joblib"
model = joblib.load(MODEL_PATH)


class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float


app = FastAPI(title="Iris inference API")

@app.post("/predict")
def predict(data: IrisRequest):
    features: List[List[float]] = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]
    pred = model.predict(features)[0]      
    return {"prediction": int(pred)}
