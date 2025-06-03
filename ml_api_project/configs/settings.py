from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST_CODE_PATH: str
    CONTAINER_CODE_PATH: str

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str

    ENV: str = "development"

    API_KEY: str

    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    class Config:
        env_file = ".env"

settings = Settings()
