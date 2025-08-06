import os
from pathlib import Path
from dotenv import load_dotenv
from model_training import LSTMModel

load_dotenv()

HOST_CODE_PATH = Path(os.getenv("HOST_CODE_PATH"))
CONTAINER_CODE_PATH = Path(os.getenv("CONTAINER_CODE_PATH"))

DATA_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'final_data.parquet'
FEATURES_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'good_features.json'

OUTPUT_DIR = HOST_CODE_PATH / 'data' / 'models'
PREDICTIONS_PATH = HOST_CODE_PATH / 'data' / 'predictions'

TARGET = "log_Quantity"

model = LSTMModel(data_path=DATA_PATH, window_size=15, features_path=FEATURES_PATH, target=TARGET, output_dir=OUTPUT_DIR)
model.output_dir = OUTPUT_DIR

cv_metrics = model.cross_validate()




