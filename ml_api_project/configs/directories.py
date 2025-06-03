import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

IS_CONTAINER = Path("/.dockerenv").exists()

BASE_PATH = Path(os.getenv("CONTAINER_CODE_PATH")) if IS_CONTAINER else Path(os.getenv("HOST_CODE_PATH"))

CAR_DATA_TRAIN = BASE_PATH / "data" / "train"
CAR_DATA_TEST = BASE_PATH / "data" / "test"
CAR_DATA_PREDICTIONS = BASE_PATH / "data" / "predictions"
CAR_DATA_MODELS = BASE_PATH / "data" / "models"