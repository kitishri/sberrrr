import logging
import os

path = os.environ.get("PROJECT_PATH", ".")
log_file = (f"{path}/configs/de_sber.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)


    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)


def log_message():
    logger.info("Airflow DAG has started.")
