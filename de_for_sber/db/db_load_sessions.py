from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from configs.logging_config import logger, send_log_to_elasticsearch

load_dotenv()

host_code_path = Path(os.getenv("HOST_CODE_PATH"))

file_path = host_code_path / 'data' / 'processed' / 'processed_sessions.pkl'

sessions = pd.read_pickle(file_path)

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

sessions.to_sql('sessions', engine, if_exists='append', index=False)
logger.info("Sessions data successfully inserted into the database")
send_log_to_elasticsearch("Sessions data successfully inserted into the database")