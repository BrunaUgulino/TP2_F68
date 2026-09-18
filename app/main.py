from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "API FastAPI fonctionnelle"}

@app.post("/predict")
def predict(data: PredictRequest):
    features = data.features
    if not features:
        raise HTTPException(status_code=400, detail="La liste features ne peut pas être vide")
    
    prediction_result = sum(features)
    return {"prediction": prediction_result}
