import logging
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import os

load_dotenv()

host_code_path = os.getenv("HOST_CODE_PATH")
if not host_code_path:
    print("HOST_CODE_PATH is not set! Using current directory.")
    host_code_path = os.getcwd()

log_file = os.path.join(host_code_path, "configs", "de_sber.log")

log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    print(f"Directory {log_dir} does not exist. Creating...")
    os.makedirs(log_dir)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

es = Elasticsearch("http://elasticsearch:9200", retry_on_timeout=True, max_retries=5)

@retry(stop=stop_after_attempt(1), wait=wait_fixed(1), retry=retry_if_exception_type(Exception))
def send_log_to_elasticsearch(level, message):
    log_entry = {
        "@timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        "service": "general_service"
    }

    try:
        es.index(index="logs", body=log_entry)
        logger.info(f"Log sent to Elasticsearch: {message}")
    except Exception as e:
        logger.error(f"Failed to send log to Elasticsearch: {e}")
        raise