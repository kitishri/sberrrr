import pandas as pd
import os
import json
import pickle
from sqlalchemy import create_engine
from configs.logging_config import logger
path = os.environ.get('PROJECT_PATH', '.')
FLAGS_FILE_HITS = (f'{path}/configs/processing_flags_hits.json')
RAW_DIR = (f'{path}/data/raw/new_hits')
PROCESSED_DIR = (f'{path}/data/processed/new_hits')
print(os.path.abspath(path))

# Функция для загрузки флажков
def load_flags_hits():

    if os.path.exists(FLAGS_FILE_HITS):
        with open(FLAGS_FILE_HITS, 'r') as file:
            return json.load(file)
    else:
        flags = {"hits_data": {}}
        save_flags_hits(flags)
        return flags

# Функция для сохранения флажков
def save_flags_hits(flags):
    with open(FLAGS_FILE_HITS, 'w') as file:
        json.dump(flags, file, indent=4)

def transform_new_files_hits():

    # Загружаем флаги
    flags = load_flags_hits()

    for file_name in os.listdir(RAW_DIR):
        file_path = os.path.join(RAW_DIR, file_name)

        file_flags = flags['hits_data'].setdefault(file_name, {
            "loaded": False,
            "transformed": False,
            "to_db": False,
            "saved": False
        })

        # Пропускаем уже загруженные файлы
        if file_flags["loaded"]:
            logger.warning(f"File {file_name} has already been loaded. Skipping.")
            continue

        with open(file_path, 'r') as file:
            data_new = json.load(file)

        all_sessions = [session for year_data in data_new.values() for session in year_data]
        if not all_sessions:
            logger.warning(f"File {file_name} contains no valid sessions. Skipping.")
            continue
        df = pd.DataFrame(all_sessions)

        # Обновляем флаги и сохраняем
        flags["hits_data"][file_name]["loaded"] = True
        save_flags_hits(flags)

        file_name_pkl = file_name.replace('.json', '.pkl')

        if flags["hits_data"].get(file_name, {}).get("loaded") and not flags["hits_data"].get(file_name_pkl, {}).get(
                "transformed"):
            logger.info(f"Transforming data for {file_name}...")

            # Проверка обязательных колонок
            required_columns = ["session_id", "hit_date", "hit_time", "hit_number", "hit_type", "hit_referer", "hit_page_path",
                                "event_category", "event_action", "event_label", "event_value"
                                ]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"File {file_path} is missing columns: {', '.join(missing_columns)}. Skipping.")
                return False

            # Очистка данных
            logger.info(f"Cleaning data for {file_name}...")
            df = df.drop(columns=['hit_time', 'event_value', 'hit_type'])

            # Заполнение пропусков
            logger.info(f"Filling missing values in {file_name}...")
            df['hit_referer'] = df['hit_referer'].fillna('unknown')
            df['event_label'] = df['event_label'].fillna('unknown')

            # Преобразуем даты
            logger.info(f"Converting 'hit_date' to datetime for {file_name}...")
            df['hit_date'] = pd.to_datetime(df['hit_date'], errors='coerce')

            # Преобразуем session_id в строки
            logger.info(f"Converting 'session_id' to string for {file_name}...")
            df["session_id"] = df["session_id"].astype(str)

            # Генерация hit_id
            logger.info(f"Generating 'hit_id' for {file_name}...")
            df['hit_id'] = df['session_id'] + '_' + df['hit_number'].astype(str)

            # Удаление дубликатов
            logger.info(f"Removing duplicates for {file_name}...")
            duplicates = df[df.duplicated(subset='hit_id', keep=False)]
            duplicates = duplicates.replace('unknown', pd.NA)
            duplicates = duplicates.sort_values(by=['hit_id', 'hit_referer', 'event_label'],
                                                ascending=[True, True, True])

            to_drop = duplicates.drop_duplicates(subset='hit_id', keep='last')
            indexes_to_drop = to_drop.index
            df_hits = df.drop(indexes_to_drop)

            # Завершающие шаги: сохраняем или возвращаем очищенные данные
            logger.info(f"Data from {file_name} transformed successfully.")

            # Сохранение данных в файл
            output_file = os.path.join(PROCESSED_DIR, file_name_pkl)
            df_hits.to_pickle(output_file)

            # Обновление флагов
            flags["hits_data"][file_name_pkl] = flags["hits_data"].pop(file_name, {})
            flags["hits_data"][file_name_pkl]["transformed"] = True
            flags["hits_data"][file_name_pkl]["saved"] = True
            save_flags_hits(flags)
            logger.info(f"Transformed data saved to file: {output_file}")

        else:
            logger.warning(f"Skipping transformation for {file_name}")

    logger.info("All files have been processed.")



def send_hits_to_db():

    flags = load_flags_hits()

    engine = create_engine("postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de")

    for file_name in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, file_name)

        if flags["hits_data"].get(file_name, {}).get("transformed") and not flags["hits_data"][file_name].get("to_db"):
            logger.info(f"Sending data for {file_name} to db...")

            # Сохранение в базу данных
            try:
                with open(file_path, 'rb') as file:
                    hits_new = pickle.load(file)

                df_hits = pd.DataFrame(hits_new)
                df_hits.to_sql("hits", con=engine, if_exists="append", index=False)
                flags["hits_data"][file_name]["to_db"] = True
                save_flags_hits(flags)
                logger.info(f"Transformed data for {file_name} added to the database.")

            except Exception as e:
                logger.error(f"Error saving data for {file_name} to the database: {e}")






