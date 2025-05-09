from pathlib import Path
from dotenv import load_dotenv
import os
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    Integer, String, ForeignKey, Date, Time
)
from configs.logging_config import logger, send_log_to_elasticsearch

load_dotenv()

# Load environment variables

host_code_path = Path(os.getenv("HOST_CODE_PATH"))

# Database connection setup
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

metadata = MetaData()

# Define sessions table
sessions_table = Table(
    'sessions', metadata,
    Column('session_id', String, primary_key=True),
    Column('client_id', String, nullable=False),
    Column('visit_date', Date, nullable=False),
    Column('visit_time', Time, nullable=False),
    Column('visit_number', Integer, nullable=False),
    Column('utm_source', String),
    Column('utm_medium', String),
    Column('utm_campaign', String),
    Column('utm_keyword', String),
    Column('utm_adcontent', String),
    Column('device_category', String),
    Column('device_os', String),
    Column('device_brand', String),
    Column('device_screen_resolution', String),
    Column('device_browser', String),
    Column('geo_country', String),
    Column('geo_city', String)
)

# Define hits table
hits_table = Table(
    'hits', metadata,
    Column('hit_id', String, primary_key=True),
    Column('session_id', String, ForeignKey('sessions.session_id'), nullable=False),
    Column('hit_date', Date, nullable=False),
    Column('hit_number', Integer, nullable=False),
    Column('hit_referer', String),
    Column('hit_page_path', String),
    Column('event_category', String),
    Column('event_action', String),
    Column('event_label', String)
)

metadata.create_all(engine)
logger.info("Database tables created successfully")
send_log_to_elasticsearch("Database tables created successfully")
