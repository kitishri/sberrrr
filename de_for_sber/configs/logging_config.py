import logging
import os
from logging.handlers import RotatingFileHandler
from elasticsearch import Elasticsearch, ElasticsearchException
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

path = os.environ.get("PROJECT_PATH", ".")
log_file = f"{path}/configs/de_sber.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

es = Elasticsearch(
    "http://elasticsearch:9200",
    retry_on_timeout=True,
    max_retries=5
)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(ElasticsearchException))
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
    except ElasticsearchException as e:
        logger.error(f"Failed to send log to Elasticsearch: {e}")
        raise

def log_message():
    msg = "Airflow DAG has started."

    logger.info(msg)
    try:
        send_log_to_elasticsearch("INFO", msg)
    except ElasticsearchException:
        logger.error("Log could not be sent to Elasticsearch.")

    logger.warning("Potential issue detected.")
    try:
        send_log_to_elasticsearch("WARNING", "Potential issue detected.")
    except ElasticsearchException:
        logger.error("Log could not be sent to Elasticsearch.")

    logger.error("Error while processing hits.")
    try:
        send_log_to_elasticsearch("ERROR", "Error while processing hits.")
    except ElasticsearchException:
        logger.error("Log could not be sent to Elasticsearch.")
