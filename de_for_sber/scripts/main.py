from configs.config import LOGS_DIR, FLAGS_FILE,LOG_FILE, CONFIGS_DIR, PROCESSED_DIR, RAW_DIR
from logging_config import logger
import os
import json
import pandas as pd
from logging_config import logger
from new_data_to_db import load_sessions, transform_sessions, validate_sessions, load_flags, save_flags



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

def main():

    load_flags()

    process_new_sessions()

if __name__ == "__main__":
    main()

