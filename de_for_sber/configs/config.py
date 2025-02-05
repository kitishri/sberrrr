import os

# Базовые директории
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

# Пути к файлам
FLAGS_FILE_HITS = os.path.join(CONFIGS_DIR, "processing_flags_hits.json")
FLAGS_FILE_SESSIONS = os.path.join(CONFIGS_DIR, "processing_flags_sessions.json")





