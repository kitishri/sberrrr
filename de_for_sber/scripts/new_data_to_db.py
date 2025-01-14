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
        flags = {
            "sessions_data": {},
            "hits_data": {}
        }
        save_flags(flags)
        return flags

# Функция для сохранения флажков
def save_flags(flags):
    with open(FLAGS_FILE, 'w') as file:
        json.dump(flags, file, indent=4)


def load_new_files(directory, flags, flag_key):
    dataframes = []
    file_paths = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)

        file_flags = flags[flag_key].get(file)
        if file_flags is None:
            file_flags = {
                "loaded": False,
                "valid": False,
                "transformed": False,
                "saved": False,
                "to_db": False
            }
            flags[flag_key][file] = file_flags

        # Загружаем только файлы, которые еще не были обработаны
        if file_flags["loaded"]:
            continue

        if file.endswith(".json") and os.path.isfile(file_path):
            logger.info(f"Loading file: {file_path}")

            # Загружаем данные из файла
            with open(file_path, 'r') as filee:
                data_new = json.load(filee)

            if not data_new:
                logger.warning(f"File {file_path} is empty. Skipping.")
                continue


            all_sessions = [session for year_data in data_new.values() for session in year_data]
            df = pd.DataFrame(all_sessions)
            dataframes.append(df)
            file_paths.append(file_path)


            flags[flag_key][file]["loaded"] = True
            save_flags(flags)


    return dataframes, file_paths


def load_sessions(flags):
    # Загружаем новые файлы для сессий
    sessions_data, file_paths = load_new_files(os.path.join(RAW_DIR, "new_sessions"), flags, "sessions_data")

    save_flags(flags)

    return sessions_data, file_paths

def validate_sessions(df, file_path):
    if df.empty:
        logger.error(f"File {file_path} is empty. Skipping.")
        return False

    # Проверка обязательных колонок
    required_columns = [
        'session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_keyword',
        'device_category', 'device_os', 'device_brand', 'device_model',
        'device_screen_resolution', 'device_browser', 'geo_country', 'geo_city'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"File {file_path} is missing columns: {', '.join(missing_columns)}. Skipping.")
        return False

    # Проверка уникальности session_id
    if df['session_id'].duplicated().any():
        logger.error(f"File {file_path} has duplicate 'session_id'. Skipping.")
        return False

    return True


def transform_sessions(df):
    # Заполнение пропусков
    for column in ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_os', 'device_brand', 'utm_source']:
        df[column] = df[column].fillna('unknown')
    logger.info(f"Columns have been transformed.")

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

