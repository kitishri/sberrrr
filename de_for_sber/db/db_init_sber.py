from sqlalchemy import create_engine, Time, Date
from sqlalchemy import MetaData, Table, Column, Integer, String, ForeignKey, DateTime

# Создание метаданных
metadata = MetaData()
# Параметры подключения
DB_NAME = "sber_de"
USER = "Ekaterina_Firsova"
PASSWORD = "376d51"
HOST = "localhost"
PORT = 5432

# Создание подключения
engine = create_engine(f"postgresql+psycopg2://{'Ekaterina_Firsova'}:{'376d51'}@{'localhost'}:{5432}/{'sber_de'}")
connection = engine.connect()
sessions_table = Table(
    'sessions', metadata,
    Column('session_id', String, primary_key=True),
    Column('client_id', String, nullable=False),
    Column('visit_date', Date, nullable=False),
    Column('visit_time', Time, nullable=False),
    Column('visit_number', Integer, nullable=False),
    Column('utm_source', String, nullable=True),
    Column('utm_medium', String, nullable=True),
    Column('utm_campaign', String, nullable=True),
    Column('utm_keyword', String, nullable=True),
    Column('device_category', String, nullable=True),
    Column('device_os', String, nullable=True),
    Column('device_brand', String, nullable=True),
    Column('device_screen_resolution', String, nullable=True),
    Column('device_browser', String, nullable=True),
    Column('geo_country', String, nullable=True),
    Column('geo_city', String, nullable=True)
)

hits_table = Table(
    'hits', metadata,
    Column('hit_id', Integer, primary_key=True, autoincrement=True),
    Column('session_id', String, ForeignKey('sessions.session_id'), nullable=False),
    Column('hit_date', Date, nullable=False),
    Column('hit_time', Time, nullable=False),
    Column('hit_number', Integer, nullable=False),
    Column('hit_type', String, nullable=True),
    Column('hit_referer', String, nullable=True),
    Column('hit_page_path', String, nullable=True),
    Column('event_category', String, nullable=True),
    Column('event_action', String, nullable=True),
    Column('event_label', String, nullable=True),
    Column('event_value', Integer, nullable=True)
)
metadata.create_all(engine)
print("Таблицы созданы успешно!")