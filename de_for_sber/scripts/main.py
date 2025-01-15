from configs.config import LOGS_DIR, FLAGS_FILE,LOG_FILE, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR
from logging_config import logger
import os
from sqlalchemy import create_engine
import pandas as pd
from logging_config import logger
from new_data_to_db import (load_sessions, load_hits, transform_sessions, transform_hits, validate_sessions, validate_hits,
                            load_flags, save_flags)



def process_new_sessions():

    flags = load_flags()
    sessions_data, file_paths = load_sessions(flags)

    def get_session_data(sessions_data, file_paths, file_name):
        return next((df for df, path in zip(sessions_data, file_paths) if os.path.basename(path) == file_name), None)

    # Валидация данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        session_data = get_session_data(sessions_data, file_paths, file_name)

        if file_name not in flags["sessions_data"] or not flags["sessions_data"][file_name]["loaded"]:
            logger.info(f"File {file_name} has not been loaded yet or is missing flags. Skipping validation.")
            continue

        if flags["sessions_data"][file_name]["valid"]:
            logger.info(f"File {file_name} has already been validated.")
            continue

        logger.info(f"Validating data for {file_name}...")
        session_data = get_session_data(sessions_data, file_paths, file_name)

        if session_data is not None and validate_sessions(session_data, file_name):
            flags["sessions_data"][file_name]["valid"] = True
            logger.info(f"File {file_name} passed validation.")
        else:
            logger.error(f"File {file_name} failed validation or is missing data. Skipping.")

    save_flags(flags)

    # Трансформация данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)

        if file_name not in flags["sessions_data"]:
            logger.error(f"File {file_name} is missing flags. Skipping transformation.")
            continue

        if flags["sessions_data"][file_name]["valid"] and not flags["sessions_data"][file_name]["transformed"]:
            logger.info(f"Transforming data for {file_name}...")
            session_data = get_session_data(sessions_data, file_paths, file_name)

            if session_data is not None:

                transformed_data = transform_sessions(session_data)

                # Сохранение в базу данных
                try:
                    DB_NAME = "sber_de"
                    USER = "Ekaterina_Firsova"
                    PASSWORD = "376d51"
                    HOST = "localhost"
                    PORT = 5432
                    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}")
                    transformed_data.to_sql("sessions", con=engine, if_exists="append", index=False)
                    logger.info(f"Transformed data for {file_name} added to the database.")
                    flags["sessions_data"][file_name]["to_db"] = True
                except Exception as e:
                    logger.error(f"Error saving data for {file_name} to the database: {e}")
                    flags["sessions_data"][file_name]["to_db"] = False
                    continue

                output_file = os.path.join(PROCESSED_DIR, 'new', f"transformed_sessions_{file_name}")
                transformed_data.to_csv(output_file, index=False)
                logger.info(f"Transformed data saved to file: {output_file}")
                flags["sessions_data"][file_name]["transformed"] = True

            else:
                logger.warning(f"File {file_name} is missing or empty. Skipping transformation.")
        else:
            logger.info(f"File {file_name} has already been transformed or been skipped.")

    save_flags(flags)

    # Сохранение данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)  # Извлекаем имя файла из пути
        if file_name not in flags["sessions_data"]:
            logger.error(f"File {file_name} is missing flags. Skipping saving.")
            continue

        if flags["sessions_data"][file_name]["transformed"] and not flags["sessions_data"][file_name]["saved"]:
            logger.info(f"Saving transformed data for {file_name}...")
            # Тут должен быть код для сохранения данных в БД или в файл
            flags["sessions_data"][file_name]["saved"] = True
            logger.info(f"File {file_name} has been saved.")
        else:
            logger.info(f"File {file_name} has already been saved or been skipped.")

    save_flags(flags)

def process_new_hits():
    flags = load_flags()
    hits_data, file_paths = load_hits(flags)

    def get_hit_data(hits_data, file_paths, file_name):
        return next((df for df, path in zip(hits_data, file_paths) if os.path.basename(path) == file_name), None)

    # Валидация данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name not in flags["hits_data"] or not flags["hits_data"][file_name]["loaded"]:
            continue  # Пропускаем, если файл не загружен или отсутствуют флаги

        if flags["hits_data"][file_name]["valid"]:
            continue  # Пропускаем, если файл уже валидирован

        logger.info(f"Validating data for {file_name}...")
        hit_data = get_hit_data(hits_data, file_paths, file_name)

        if hit_data is not None and validate_hits(hit_data, file_name):
            flags["hits_data"][file_name]["valid"] = True
            logger.info(f"File {file_name} passed validation.")
        else:
            logger.error(f"File {file_name} failed validation or is missing data.")

    save_flags(flags)

    # Трансформация данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name not in flags["hits_data"]:
            continue  # Пропускаем, если отсутствуют флаги

        file_flags = flags["hits_data"][file_name]
        if not file_flags["valid"] or file_flags["transformed"]:
            continue  # Пропускаем, если файл не валидирован или уже трансформирован

        logger.info(f"Transforming data for {file_name}...")
        hit_data = get_hit_data(hits_data, file_paths, file_name)

        if hit_data is not None:
            transformed_data = transform_hits(hit_data)

            # Сохранение в базу данных
            if not file_flags.get("to_db", False):
                try:
                    engine = create_engine(
                        f"postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de"
                    )
                    transformed_data.to_sql("hits", con=engine, if_exists="append", index=False)
                    file_flags["to_db"] = True
                    logger.info(f"Transformed data for {file_name} added to the database.")
                except Exception as e:
                    logger.error(f"Error saving data for {file_name} to the database: {e}")
                    continue



            # Сохранение трансформированных данных в файл
            output_file = os.path.join(PROCESSED_DIR, 'new', f"transformed_hits_{file_name}")
            transformed_data.to_csv(output_file, index=False)
            file_flags["transformed"] = True
            logger.info(f"Transformed data saved to file: {output_file}")
        else:
            logger.warning(f"File {file_name} is missing or empty. Skipping transformation.")

    save_flags(flags)

    # Финальное сохранение данных
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name not in flags["hits_data"]:
            continue  # Пропускаем, если отсутствуют флаги

        file_flags = flags["hits_data"][file_name]
        if file_flags["transformed"] and not file_flags["saved"]:
            logger.info(f"Marking {file_name} as saved.")
            file_flags["saved"] = True

    save_flags(flags)

def main():

    load_flags()

    process_new_sessions()

    process_new_hits()

if __name__ == "__main__":
    main()

