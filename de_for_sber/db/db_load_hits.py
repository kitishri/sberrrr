import pandas as pd
from sqlalchemy import create_engine



# Параметры подключения
DB_NAME = "sber_de"
USER = "Ekaterina_Firsova"
PASSWORD = "376d51"
HOST = "localhost"
PORT = 5432

# Создание подключения
engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}")

# Загрузка данных
hits = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/df_sessions_filtered.csv")


# Загрузка данных в таблицу hits
hits.to_sql('hits', engine, if_exists='append', index=False, chunksize=10000)
print("Данные из hits успешно загружены!")

