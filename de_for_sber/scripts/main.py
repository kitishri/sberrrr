from configs.config import LOGS_DIR, FLAGS_FILE,LOG_FILE, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR
from logging_config import logger
import os
import json
import pandas as pd
from logging_config import logger
from new_data_to_db import load_sessions, load_hits, transform_sessions, validate_sessions, load_flags, save_flags



# Основная функция обработки sessions
def process_new_sessions():

    flags = load_flags()

    # Проверка и загрузка данных
    sessions_data, file_paths = load_sessions(flags)

    # Проверка валидации
    if flags["sessions_data"]["data_loaded"] and not flags["sessions_data"]["data_valid"]:
        logger.info("Validating data for sessions...")
        for i, (session_data, file_path) in enumerate(zip(sessions_data, file_paths)):
            logger.info(f"Validating file: {file_path}")

            if session_data.empty:
                logger.error(f"File {file_path} is empty. Skipping.")
                continue  # Пропускаем пустой файл

            if 'session_id' not in session_data.columns:
                logger.error(f"File {file_path} is missing 'session_id' column. Skipping.")
                continue  # Пропускаем файл, если нет 'session_id'

                # Выполнение валидации
            if not validate_sessions(session_data):
                logger.error(f"Validation failed for file: {file_path}. Skipping.")
                continue

            flags["sessions_data"]["data_valid"] = True
            save_flags(flags)
    else:
        logger.info("Data for sessions has already been validated.")

    # Трансформация данных
    if flags["sessions_data"]["data_valid"] and not flags["sessions_data"]["data_transformed"]:
        logger.info("Transforming data for sessions...")
        for i, (session_data, file_path) in enumerate(zip(sessions_data, file_paths)):
            if session_data.empty:
                logger.warning(f"File {file_path} is empty. Skipping transformation.")
                continue  # Пропускаем пустые данные

            logger.info(f"Transforming file: {file_path}")
            transformed_data = transform_sessions(session_data)  # Трансформация
            output_file = os.path.join(PROCESSED_DIR, f"transformed_sessions_{i + 1}.csv")
            transformed_data.to_csv(output_file, index=False)
            logger.info(f"Transformed data saved to file: {output_file}")

        flags["sessions_data"]["data_transformed"] = True
        save_flags(flags)
    else:
        logger.info("Data for sessions has already been transformed.")

    # Сохранение данных
    if flags["sessions_data"]["data_transformed"] and not flags["sessions_data"]["data_saved"]:
        logger.info("Saving transformed data for sessions...")
        flags["sessions_data"]["data_saved"] = True
        save_flags(flags)
    else:
        logger.info("Transformed data for sessions has already been saved.")



def main():

    load_flags()

    process_new_sessions()

if __name__ == "__main__":
    main()

