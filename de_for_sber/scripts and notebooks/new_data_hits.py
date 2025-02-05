import pandas as pd
import os
import json
import pickle
from sqlalchemy import create_engine
from configs.config import LOGS_DIR, FLAGS_FILE_HITS, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR
from logging_config import logger



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

def load_new_files_hits():
    dataframes_hits = []
    file_paths = []
    new_hits_dir = os.path.join(RAW_DIR, "new_hits")
    # Загружаем флаги
    flags = load_flags_hits()


    for file_name in os.listdir(new_hits_dir):
        file_path = os.path.join(new_hits_dir, file_name)



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
        dataframes_hits.append(df)
        file_paths.append(file_path)

        # Обновляем флаги и сохраняем
        flags["hits_data"][file_name]["loaded"] = True
        save_flags_hits(flags)

    return dataframes_hits, file_paths

def transform_hits(dataframes_hits, file_paths):

    flags = load_flags_hits()

    for df, file_path in zip(dataframes_hits, file_paths):
        logger.info(f"Transforming data from {file_path}")
        file_name = os.path.basename(file_path)
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
            output_file = os.path.join(PROCESSED_DIR, 'new_hits', file_name_pkl)
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
    proc_hits_dir = os.path.join(PROCESSED_DIR, "new_hits")

    flags = load_flags_hits()

    engine = create_engine("postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de")

    for file_name in os.listdir(proc_hits_dir):
        file_path = os.path.join(proc_hits_dir, file_name)

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






