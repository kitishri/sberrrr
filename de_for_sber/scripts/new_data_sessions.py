import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
from sqlalchemy import create_engine
from configs.logging_config import logger, send_log_to_elasticsearch
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import psycopg2
from prometheus_client import CollectorRegistry, Counter, push_to_gateway
import argparse

path = os.environ.get('PROJECT_PATH', '.')
if path == '.':
    logger.warning("PROJECT_PATH is not set. Using current directory as default.")

FLAGS_FILE_SESSIONS = os.path.join(path, 'configs', 'processing_flags_sessions.json')
RAW_DIR = os.path.join(path, 'data', 'raw', 'new_sessions')
PROCESSED_DIR = os.path.join(path, 'data', 'processed', 'new_sessions')

# Prometheus metrics
registry = CollectorRegistry()
sessions_transform_success = Counter('sessions_transform_success_total', 'Successful transforms', registry=registry)
sessions_transform_fail = Counter('sessions_transform_fail_total', 'Failed transforms', registry=registry)
sessions_db_success = Counter('sessions_db_success_total', 'Successful DB loads', registry=registry)
sessions_db_fail = Counter('sessions_db_fail_total', 'Failed DB loads', registry=registry)


def log_operation(level: str, message: str, es_required: bool = False):
    try:
        if level == 'INFO':
            logger.info(message)
        elif level == 'WARNING':
            logger.warning(message)
        elif level == 'ERROR':
            logger.error(message)
        elif level == 'DEBUG':
            logger.debug(message)

        if level in ('ERROR', 'WARNING') or es_required:
            try:
                send_log_to_elasticsearch(level, message)
            except Exception as es_error:
                logger.error(f"Failed to send log to Elasticsearch: {str(es_error)}")
    except Exception as log_error:
        print(f"CRITICAL LOGGING FAILURE: {str(log_error)} | Original message: {message}")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(OSError))
def load_flags_sessions():
    try:
        if os.path.exists(FLAGS_FILE_SESSIONS):
            with open(FLAGS_FILE_SESSIONS, 'r') as file:
                log_operation('DEBUG', f"Loading flags from {FLAGS_FILE_SESSIONS}")
                return json.load(file)
        else:
            log_operation('INFO', "Flags file not found, creating new one")
            flags = {"sessions_data": {}}
            save_flags_sessions(flags)
            return flags
    except Exception as e:
        log_operation('ERROR', f"Failed to load flags: {str(e)}")
        raise

def save_flags_sessions(flags):
    try:
        with open(FLAGS_FILE_SESSIONS, 'w') as file:
            json.dump(flags, file, indent=4)
            log_operation('DEBUG', "Flags successfully saved")
    except Exception as e:
        log_operation('ERROR', f"Error saving flags: {str(e)}")
        raise

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(ValueError))
def transform_new_files_sessions():
    if not os.path.exists(RAW_DIR):
        log_operation('ERROR', f"Directory {RAW_DIR} does not exist.")
        return

    flags = load_flags_sessions()

    for file_name in os.listdir(RAW_DIR):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(RAW_DIR, file_name)
        file_name_pkl = file_name.replace('.json', '.pkl')

        file_flags = flags['sessions_data'].setdefault(file_name, {
            "loaded": False,
            "transformed": False,
            "to_db": False,
            "saved": False
        })

        if file_flags["loaded"]:
            log_operation('WARNING', f"File {file_name} has already been loaded. Skipping.")
            continue

        try:
            with open(file_path, 'r') as file:
                data_new = json.load(file)
        except FileNotFoundError:
            log_operation('ERROR', f"File {file_name} not found. Retrying...")
            raise

        all_sessions = [session for year_data in data_new.values() for session in year_data]
        if not all_sessions:
            log_operation('WARNING', f"File {file_name} contains no valid sessions. Skipping.")
            continue

        df = pd.DataFrame(all_sessions)
        file_flags["loaded"] = True
        save_flags_sessions(flags)

        if not file_flags["transformed"]:
            try:
                log_operation('DEBUG', "Cleaning data...")

                req_cols = ['session_id','client_id','visit_date','visit_time','visit_number','utm_source','utm_medium','utm_campaign','utm_adcontent','utm_keyword','device_category','device_os','device_brand','device_screen_resolution','device_browser','geo_country','geo_city']
                missing = [c for c in req_cols if c not in df.columns]
                if missing:
                    log_operation('ERROR', f"Missing columns: {missing}.")
                    sessions_transform_fail.inc()
                    continue

                for col in ['utm_campaign','utm_adcontent','utm_keyword','device_os','device_brand','utm_source']:
                    df[col] = df[col].fillna('unknown')
                df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
                df['client_id'] = df['client_id'].astype(str)
                df['session_id']= df['session_id'].astype(str)
                df.drop(columns='device_model', inplace=True, errors='ignore')
                df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S', errors='coerce').dt.time

                file_name_pkl = file_name.replace('.json', '.pkl')
                output_file = os.path.join(PROCESSED_DIR, file_name_pkl)

                file_flags["transformed"] = True
                file_flags["saved"] = True
                save_flags_sessions(flags)

                sessions_transform_success.inc()
                log_operation('INFO', f"Transformed data saved to {output_file}")

            except Exception as e:
                sessions_transform_fail.inc()
                log_operation('ERROR', f"Transform error for {file_name}: {e}")
                raise
        else:
            log_operation('WARNING', f"Skipping transformation for {file_name}")

    log_operation('INFO', "All files have been processed.")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(psycopg2.OperationalError))
def send_sessions_to_db():
    if not os.path.exists(PROCESSED_DIR):
        log_operation('ERROR', f"Directory {PROCESSED_DIR} does not exist.")
        return

    flags = load_flags_sessions()

    engine = create_engine("postgresql+psycopg2://Ekaterina_Firsova:376d51@localhost:5432/sber_de")

    for file_name in os.listdir(PROCESSED_DIR):
        file_path = os.path.join(PROCESSED_DIR, file_name)

        if file_name not in flags["hits_data"]:
            continue

        file_flags = flags["hits_data"][file_name]
        if not file_flags.get("transformed") or file_flags.get("to_db"):
            continue
        try:
            log_operation('INFO', f"Sending {file_name} to DB...")

            with open(file_path, 'rb') as f:
                df = pd.DataFrame(pickle.load(f))
            df.to_sql('sessions', con=engine, if_exists='append', index=False)

            flags['sessions_data'][file_name]['to_db'] = True
            save_flags_sessions(flags)

            sessions_db_success.inc()

            log_operation('INFO', f"Data from {file_name} loaded to DB.")

        except Exception as e:
            sessions_db_fail.inc()

            log_operation('ERROR', f"DB load error for {file_name}: {e}")
            raise


if __name__ == '__main__':
    try:
        log_operation('INFO', "ETL process started", es_required=True)

        parser = argparse.ArgumentParser(description='Process sessions data.')
        parser.add_argument('--process', action='store_true', help='Transform raw data')
        parser.add_argument('--to_db', action='store_true', help='Load to PostgreSQL')
        args = parser.parse_args()

        if args.process:
            transform_new_files_sessions()
        if args.to_db:
            send_sessions_to_db()


        push_to_gateway('localhost:9091', job='sessions_processing_job', registry=registry)
        log_operation('INFO', "ETL process completed successfully", es_required=True)

    except Exception as e:
        log_operation('ERROR',
                          f"ETL process crashed: {str(e)}",
                          es_required=True)
        raise
