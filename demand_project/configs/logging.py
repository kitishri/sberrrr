import sys
import os
import time
from functools import wraps
from loguru import logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

def log_step(name=None):
    def decorator(func):
        step_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting step: {step_name}")
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"Completed step: {step_name} in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator
