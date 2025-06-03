from app.celery_app import celery_app
from datetime import datetime
import pandas as pd
import dill
import json
from configs.directories import CAR_DATA_TEST,CAR_DATA_MODELS, CAR_DATA_PREDICTIONS

model_path = max(CAR_DATA_MODELS.glob("*.pkl"), key=lambda f: f.stat().st_mtime)

with open(model_path, "rb") as f:
    model = dill.load(f)

@celery_app.task(name="app.tasks.predict_all")
def predict_all_task():
    json_files = list(CAR_DATA_TEST.glob("*.json"))
    if not json_files:
        return {"error": "No JSON files found in test data folder"}

    all_predictions = {}

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        df = pd.DataFrame([data_dict])
        preds = model.predict(df)
        all_predictions[file_path.name] = preds.tolist()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_file = CAR_DATA_PREDICTIONS / f"{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"predictions": all_predictions}, f, ensure_ascii=False, indent=2)

    return {"message": f"Predictions saved to {output_file.name}", "predictions": all_predictions}