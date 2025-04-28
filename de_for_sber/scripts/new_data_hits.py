import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
from sqlalchemy import create_engine
from configs.logging_config import logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import psycopg2
from prometheus_client import CollectorRegistry, Counter, push_to_gateway


path = os.environ.get('PROJECT_PATH', '.')
if path == '.':
    logger.warning("PROJECT_PATH is not set. Using current directory as default.")

FLAGS_FILE_HITS = os.path.join(path, 'configs', 'processing_flags_hits.json')
RAW_DIR = os.path.join(path, 'data', 'raw', 'new_hits')
PROCESSED_DIR = os.path.join(path, 'data', 'processed', 'new_hits')

registry = CollectorRegistry()
hits_transform_success = Counter('hits_transform_success_total', 'Hits transform success', registry=registry)
hits_transform_fail = Counter('hits_transform_fail_total', 'Hits transform fails', registry=registry)
hits_db_success = Counter('hits_db_success_total', 'Hits successfully sent to DB', registry=registry)
hits_db_fail = Counter('hits_db_fail_total', 'Hits failed to send to DB', registry=registry)

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(OSError))
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
    try:
        with open(FLAGS_FILE_HITS, 'w') as file:
            json.dump(flags, file, indent=4)
    except Exception as e:
        logger.error(f"Error saving flags: {e}")
        raise

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(ValueError))
def transform_new_files_hits():
    if not os.path.exists(RAW_DIR):
        logger.error(f"Directory {RAW_DIR} does not exist.")
        return

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

        try:
            with open(file_path, 'r') as file:
                data_new = json.load(file)
        except FileNotFoundError:
            logger.error(f"File {file_name} not found. Retrying...")
            raise

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
            try:
                logger.info(f"Transforming data for {file_name}...")

                # Проверка обязательных колонок
                required_columns = ["session_id", "hit_date", "hit_time", "hit_number", "hit_type", "hit_referer", "hit_page_path",
                                    "event_category", "event_action", "event_label", "event_value"
                                    ]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    logger.error(f"File {file_path} is missing columns: {', '.join(missing_columns)}. Skipping.")
                    continue

                logger.info(f"Cleaning data for {file_name}...")

                # Удаление ненужных колонок
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
                duplicates = duplicates.sort_values(by=['hit_id', 'hit_referer', 'event_label'], ascending=[True, True, True])

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

                hits_transform_success.inc()
                logger.info(f"Transformed data saved to file: {output_file}")

            except Exception as e:
                hits_transform_fail.inc()
                logger.error(f"Error: {e}")
                raise

        else:
            logger.warning(f"Skipping transformation for {file_name}")

    logger.info("All files have been processed.")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(psycopg2.OperationalError))
def send_hits_to_db():
    if not os.path.exists(PROCESSED_DIR):
        logger.error(f"Directory {PROCESSED_DIR} does not exist.")
        return

    flags = load_flags_hits()
    engine = create_engine("postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de")

    for file_name in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, file_name)

        if flags["hits_data"].get(file_name, {}).get("transformed") and not flags["hits_data"][file_name].get("to_db"):

            # Сохранение в базу данных
            try:
                logger.info(f"Sending data for {file_name} to db...")
                with open(file_path, 'rb') as file:
                    hits_new = pickle.load(file)

                df_hits = pd.DataFrame(hits_new)
                df_hits.to_sql("hits", con=engine, if_exists="append", index=False)

                flags["hits_data"][file_name]["to_db"] = True
                save_flags_hits(flags)

                hits_db_success.inc()
                logger.info(f"Transformed data for {file_name} added to the database.")

            except Exception as e:
                hits_db_fail.inc()
                logger.error(f"Error saving data for {file_name} to the database: {e}")
                raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform and send to db")
    parser.add_argument("--process", action="store_true", help="Transforming")
    parser.add_argument("--to_db", action="store_true", help="SEnding to db")

    args = parser.parse_args()

    if args.process:
        transform_new_files_hits()

    if args.to_db:
        send_hits_to_db()

    push_to_gateway('localhost:9091', job='hits_processing_job', registry=registry)
    logger.info("Metrics pushed to Pushgateway")