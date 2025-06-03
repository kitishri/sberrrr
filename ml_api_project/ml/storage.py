import os
import boto3
from dotenv import load_dotenv
from configs.directories import CAR_DATA_MODELS

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
)

BUCKET = os.getenv("MINIO_BUCKET")


def upload_model(local_filename: str, remote_name: str = None):
    local_path = CAR_DATA_MODELS / local_filename
    remote_key = remote_name or local_filename
    s3.upload_file(str(local_path), BUCKET, remote_key)
    print(f"✅ Uploaded {local_path} to MinIO bucket '{BUCKET}' as '{remote_key}'")


def download_model(remote_name: str, local_filename: str):
    local_path = CAR_DATA_MODELS / local_filename
    s3.download_file(BUCKET, remote_name, str(local_path))
    print(f"✅ Downloaded '{remote_name}' from MinIO bucket '{BUCKET}' to '{local_path}'")
