import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

IS_CONTAINER = Path("/.dockerenv").exists()

BASE_PATH = Path(os.getenv("CONTAINER_CODE_PATH")) if IS_CONTAINER else Path(os.getenv("HOST_CODE_PATH"))

DATA_RAW_SESSIONS = BASE_PATH / "data" / "raw" / "new_sessions"
DATA_PROCESSED_SESSIONS = BASE_PATH / "data" / "processed" / "new_sessions"
FLAGS_FILE_SESSIONS = BASE_PATH / "configs" / "processing_flags_sessions.json"

DATA_RAW_HITS = BASE_PATH / "data" / "raw" / "new_hits"
DATA_PROCESSED_HITS = BASE_PATH / "data" / "processed" / "new_hits"
FLAGS_FILE_HITS = BASE_PATH / "configs" / "processing_flags_hits.json"