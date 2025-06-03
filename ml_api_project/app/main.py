from configs.settings import settings
from configs.directories import CAR_DATA_MODELS
from celery.result import AsyncResult
from app.tasks.tasks import predict_all_task
from app.celery_app import celery_app
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
import os
import dill
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
def verify_api_key(x_api_key: str = Header(..., alias="X-API-KEY")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

MODEL_PATH = sorted(CAR_DATA_MODELS.glob("cars_pipe_*.pkl"))[-1]
with open(MODEL_PATH, "rb") as f:
    model = dill.load(f)

app = FastAPI(
    title="Car Price Prediction API",
    docs_url=None if settings.ENV == "production" else "/docs",
    redoc_url=None if settings.ENV == "production" else "/redoc",
    openapi_url=None if settings.ENV == "production" else "/openapi.json",
)

class CarFeatures(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str

@app.get('/status')
def status():
    return {"status": "I'm ok"}

@app.post("/predict-all")
def start_predict_all(api_key: str = Depends(verify_api_key)):
    task = predict_all_task.delay()
    return {"task_id": task.id}

from fastapi import Depends

@app.get("/result/{task_id}")
def get_result(task_id: str, api_key: str = Depends(verify_api_key)):
    result = AsyncResult(task_id, app=celery_app)

    if result.state == "PENDING":
        return {"status": "pending"}

    if result.state == "STARTED":
        return {"status": "started"}

    if result.state == "SUCCESS":
        return {"status": "success", "result": result.result}

    if result.state == "FAILURE":
        return {
            "status": "failure",
            "error": str(result.result),
            "traceback": result.traceback,
        }

    return {"status": result.state}
