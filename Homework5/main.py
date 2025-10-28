from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the prebuilt model from the base image
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(client: Client):
    X = [client.dict()]
    probability = model.predict_proba(X)[0, 1]
    return {"probability": float(probability)}
