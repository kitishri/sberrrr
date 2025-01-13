import pandas as pd
import logging
import os
import json
from sqlalchemy import create_engine
from configs.config import LOGS_DIR, FLAGS_FILE,LOG_FILE, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR
from logging_config import logger

# Параметры подключения
DB_NAME = "sber_de"
USER = "Ekaterina_Firsova"
PASSWORD = "376d51"
HOST = "localhost"
PORT = 5432

# Создание подключения
engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}")

# Функция для загрузки флажков
def load_flags():
    if os.path.exists(FLAGS_FILE):
        with open(FLAGS_FILE, 'r') as file:
            return json.load(file)
    else:
        # Если файла нет, создаем структуру с начальными флажками
        flags = {
            "hits_data": {
                "data_loaded": False,
                "data_valid": False,
                "data_transformed": False,
                "data_saved": False
            },
            "sessions_data": {
                "data_loaded": False,
                "data_valid": False,
                "data_transformed": False,
                "data_saved": False
            }
        }
        save_flags(flags)
        return flags

# Функция для сохранения флажков
def save_flags(flags):
    with open(FLAGS_FILE, 'w') as file:
        json.dump(flags, file, indent=4)


def load_new_files(directory):
    dataframes = []
    file_paths = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".json") and os.path.isfile(file_path):
            logger.info(f"Loading file: {file_path}")
            with open(file_path, 'r') as filee:
                data_new = json.load(filee)
            all_sessions = []
            for year_data in data_new.values():
                all_sessions.extend(year_data)

            df = pd.DataFrame(all_sessions)

            dataframes.append(df)
            file_paths.append(file_path)

            print(f"File {file} loaded, {df.shape[0]} rows.")
        else:
            print(f"File {file} is not a JSON or does not exist.")
    return dataframes, file_paths


# Загрузка данных для sessions
def load_sessions(flags):
    sessions_data = []
    if not flags["sessions_data"]["data_loaded"]:
        logger.info("Loading data for sessions...")
        sessions_data, file_paths = load_new_files(os.path.join(RAW_DIR, "new_sessions"))  # Загрузка файлов для sessions
        flags["sessions_data"]["data_loaded"] = True
        save_flags(flags)
    else:
        logger.info("Data for sessions has already been loaded.")
    return sessions_data, file_paths

# Загрузка данных для hits
def load_hits(flags):
    hits_data = []
    if not flags["hits_data"]["data_loaded"]:
        logger.info("Loading data for hits...")
        hits_data = load_new_files(os.path.join(RAW_DIR, "new_hits"))  # Загрузка файлов для hits
        flags["hits_data"]["data_loaded"] = True
        save_flags(flags)
    else:
        logger.info("Data for hits has already been loaded.")
    return hits_data


def validate_sessions(df):

    if df.empty:
        logger.error("DataFrame is empty")
        return
    # Проверка обязательных колонок

    required_columns = ['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
                        'utm_source', 'utm_medium', 'utm_campaign', 'utm_keyword',
                        'device_category', 'device_os', 'device_brand', 'device_model',
                        'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city']

    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return


        # Проверка уникальности session_id
    if 'session_id' not in df.columns:
        logger.error("Missing 'session_id'")
        return
    if df['session_id'].duplicated().any():
        logger.error("Duplicate session_id found")
        return
    else:
        logger.info("All session_id are unique.")


def transform_sessions(df):
    # Заполнение пропусков
    for column in ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_os', 'device_brand', 'utm_source']:
        df[column] = df[column].fillna('unknown')
        logger.info(f"Column {column} has been transformed.")

    # Преобразование дат
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors='coerce')
    logger.info(f"Date has been transformed.")

    # Преобразование времени
    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S').dt.time
    logger.info(f"Time has been transformed.")

    # Преобразование типов данных
    df["client_id"] = df["client_id"].astype(str)
    logger.info(f"Client ID has been transformed.")

    # Удаление ненужной колонки
    df = df.drop(columns='device_model', axis=1)
    logger.info(f"Device Model has been transformed.")

    return df

