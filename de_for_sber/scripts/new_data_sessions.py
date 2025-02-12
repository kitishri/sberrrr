import pandas as pd
import os
import json
from sqlalchemy import create_engine
from configs.logging_config import logger
import pickle
path = os.environ.get('PROJECT_PATH', '.')
FLAGS_FILE_SESSIONS = (f'{path}/configs/processing_flags_sessions.json')
RAW_DIR = f'{path}/data/raw/new_sessions'
PROCESSED_DIR = f'{path}/data/processed/new_sessions'
print("Текущая рабочая директория:", os.getcwd())


# Функция для загрузки флажков
def load_flags_sessions():

    if os.path.exists(FLAGS_FILE_SESSIONS):
        with open(FLAGS_FILE_SESSIONS, 'r') as file:
            return json.load(file)
    else:
        flags = {"sessions_data": {}}
        save_flags_sessions(flags)
        return flags

# Функция для сохранения флажков
def save_flags_sessions(flags):
    with open(FLAGS_FILE_SESSIONS, 'w') as file:
        json.dump(flags, file, indent=4)

def transform_new_files_sessions():

    # Загружаем флаги
    flags = load_flags_sessions()

    for file_name in os.listdir(RAW_DIR):

        file_path = os.path.join(RAW_DIR, file_name)

        # Если файла нет в флагах — создаем начальные значения
        file_flags = flags['sessions_data'].setdefault(file_name, {
            "loaded": False,
            "transformed": False,
            "to_db": False,
            "saved": False
        })

        # Пропускаем уже загруженные файлы
        if file_flags["loaded"]:
            logger.info(f"File {file_name} has already been loaded. Skipping.")
            continue

        with open(file_path, 'r') as file:
            data_new = json.load(file)

        all_sessions = [session for year_data in data_new.values() for session in year_data]
        if not all_sessions:
            logger.warning(f"File {file_name} contains no valid sessions. Skipping.")
            continue
        df = pd.DataFrame(all_sessions)

        # Обновляем флаги и сохраняем
        flags["sessions_data"][file_name]["loaded"] = True
        save_flags_sessions(flags)

        file_name_pkl = file_name.replace('.json', '.pkl')


        if flags["sessions_data"].get(file_name, {}).get("loaded") and not flags["sessions_data"].get(file_name_pkl, {}).get(
            "transformed"):

            # Проверка обязательных колонок
            required_columns = ['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number',
       'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
       'utm_keyword', 'device_category', 'device_os', 'device_brand',
       'device_screen_resolution', 'device_browser', 'geo_country',
       'geo_city'
                                ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"File {file_path} is missing columns: {', '.join(missing_columns)}. Skipping.")
                return False

            # Заполнение пропусков
            logger.info(f"Filling missing values in {file_name}...")

            for column in ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_os', 'device_brand', 'utm_source']:
                df[column] = df[column].fillna('unknown')

            # Преобразование даты
            logger.info(f"Converting 'visit_date' to datetime for {file_name}...")
            df["visit_date"] = pd.to_datetime(df["visit_date"], errors='coerce')

            # Преобразование типов данных
            logger.info(f"Converting to string for {file_name}...")
            df["client_id"] = df["client_id"].astype(str)
            df["session_id"] = df["session_id"].astype(str)

            # Удаление пустой колонки
            logger.info(f"Dropping Device_model column for {file_name}...")
            df = df.drop(columns='device_model', axis=1)

            # Преобразование времени
            logger.info(f"Converting 'visit_time' to datetime for {file_name}...")
            df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S').dt.time

            logger.info(f"Data from {file_name} transformed successfully.")

            # Сохранение данных в файл
            output_file = os.path.join(PROCESSED_DIR, file_name_pkl)
            df.to_pickle(output_file)

            # Обновление флагов
            flags["sessions_data"][file_name_pkl] = flags["sessions_data"].pop(file_name, {})
            flags["sessions_data"][file_name_pkl]["transformed"] = True
            flags["sessions_data"][file_name_pkl]["saved"] = True
            save_flags_sessions(flags)
            logger.info(f"Transformed data saved to file: {output_file}")

        else:
            logger.warning(f"Skipping transformation for {file_name}")

        logger.info("All files have been processed.")

def send_sessions_to_db():

    flags = load_flags_sessions()

    engine = create_engine("postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de")

    for file_name in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, file_name)

        if flags["sessions_data"].get(file_name, {}).get("transformed") and not flags["sessions_data"][file_name].get("to_db"):
            logger.info(f"Sending data for {file_name} to db...")

            # Сохранение в базу данных
            try:
                with open(file_path, 'rb') as file:
                    df = pickle.load(file)

                df = pd.DataFrame(df)
                df.to_sql("sessions", con=engine, if_exists="append", index=False)
                flags["sessions_data"][file_name]["to_db"] = True
                save_flags_sessions(flags)
                logger.info(f"Transformed data for {file_name} added to the database.")

            except Exception as e:
                logger.error(f"Error saving data for {file_name} to the database: {e}")