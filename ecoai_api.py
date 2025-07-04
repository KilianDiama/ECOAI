from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ecoai import EcoAI  # Ton module existant

app = FastAPI(title="EcoAI API", description="Green & Secure NLP Toolkit API")

eco = EcoAI(master_password="supersecur3password", business_name="EcoAI API")

# --- Models
class TextInput(BaseModel):
    text: str
    user_id: str = "public"

class BatchInput(BaseModel):
    texts: List[str]
    user_id: str = "public"

# --- Endpoints
@app.post("/predict")
def predict(input: TextInput):
    try:
        result = eco.predict(input.text, user_id=input.user_id)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
def cluster(input: BatchInput):
    try:
        predictions = eco.batch_predict(input.texts, user_id=input.user_id)
        features = eco.get_embeddings(input.texts)
        reduced, clusters, score = eco.reduce_and_cluster(features)
        return {
            "clusters": clusters.tolist(),
            "score": round(score, 3),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
def get_summary():
    total_time = sum(d for d, _ in eco.energy_log)
    total_emissions = sum(e for _, e in eco.energy_log)
    return {
        "total_prediction_time_sec": round(total_time, 2),
        "total_CO2_emissions_kg": round(total_emissions, 6),
        "cached_predictions": len(eco.cache)
    }
