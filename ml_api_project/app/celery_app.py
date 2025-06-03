from celery import Celery
import os

celery_app = Celery(
    "ml_app",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
)

celery_app.conf.task_routes = {
    "app.tasks.*": {"queue": "ml_tasks"},
}

import app.tasks.tasks