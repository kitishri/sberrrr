import pandas as pd
import os
from dotenv import load_dotenv
import json
import pickle
from sqlalchemy import create_engine
from configs.logging_config import logger, send_log_to_elasticsearch
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import psycopg2
from prometheus_client import CollectorRegistry, Counter, push_to_gateway
import argparse
from configs.redis_conn import get_redis_connection
from configs.directories import FLAGS_FILE_HITS, DATA_RAW_HITS, DATA_PROCESSED_HITS
load_dotenv()

def log_operation(level, message, es_required=False):
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
                logger.error(f"Failed to send to ES: {str(es_error)}")
    except Exception as log_error:
        print(f"CRITICAL LOG FAIL: {str(log_error)} | Message: {message}")

registry = CollectorRegistry()
hits_transform_success = Counter('hits_transform_success_total', 'Hits transform success', registry=registry)
hits_transform_fail = Counter('hits_transform_fail_total', 'Hits transform fails', registry=registry)
hits_db_success = Counter('hits_db_success_total', 'Hits successfully sent to DB', registry=registry)
hits_db_fail = Counter('hits_db_fail_total', 'Hits failed to send to DB', registry=registry)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(OSError))
def load_flags_hits():
    try:
        raw_data = get_redis_connection().get('flags:hits_data')
        if raw_data:
            log_operation('DEBUG', "Flags loaded from Redis")
            return json.loads(raw_data)

        elif os.path.exists(FLAGS_FILE_HITS):
            with open(FLAGS_FILE_HITS, 'r') as file:
                log_operation('DEBUG', "Flags loaded from file (Redis empty)")
                flags = json.load(file)

                get_redis_connection().set('flags:hits_data', json.dumps(flags))

                return flags

        else:
            log_operation('INFO', "Flags not found, initializing empty")
            flags = {"hits_data": {}}
            save_flags_hits(flags)
            return flags

    except Exception as e:
        log_operation('ERROR', f"Failed to load flags: {str(e)}")
        raise

def save_flags_hits(flags):
    try:
        get_redis_connection().set('flags:hits_data', json.dumps(flags))

        with open(FLAGS_FILE_HITS, 'w') as file:
            json.dump(flags, file, indent=4)

        log_operation('DEBUG', "Flags saved to Redis and file")

    except Exception as e:
        log_operation('ERROR', f"Error saving flags: {e}")
        raise

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(ValueError))
def transform_new_files_hits():
    if not os.path.exists(DATA_RAW_HITS):
        log_operation('ERROR', f"Directory {DATA_RAW_HITS} does not exist.")
        return

    flags = load_flags_hits()

    for file_name in os.listdir(DATA_RAW_HITS):

        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(DATA_RAW_HITS, file_name)
        file_name_pkl = file_name.replace('.json', '.pkl')

        file_flags = flags['hits_data'].setdefault(file_name_pkl, {
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
        save_flags_hits(flags)

        if not file_flags["transformed"]:
            try:
                log_operation('DEBUG', "Cleaning data...")

                required_columns = [
                    "session_id", "hit_date", "hit_time", "hit_number",
                    "hit_type", "hit_referer", "hit_page_path",
                    "event_category", "event_action", "event_label", "event_value"
                ]
                missing = [c for c in required_columns if c not in df.columns]
                if missing:
                    log_operation('ERROR', f"Missing columns: {missing}.")
                    hits_transform_fail.inc()
                    continue

                df = df.drop(columns=['hit_time', 'event_value', 'hit_type'])
                df['hit_referer'] = df['hit_referer'].fillna('unknown')
                df['event_label'] = df['event_label'].fillna('unknown')
                df['hit_date'] = pd.to_datetime(df['hit_date'], errors='coerce')
                df["session_id"] = df["session_id"].astype(str)
                df['hit_id'] = df['session_id'] + '_' + df['hit_number'].astype(str)

                duplicates = df[df.duplicated(subset='hit_id', keep=False)]
                duplicates = duplicates.replace('unknown', pd.NA)
                duplicates = duplicates.sort_values(by=['hit_id', 'hit_referer', 'event_label'], ascending=[True, True, True])

                to_drop = duplicates.drop_duplicates(subset='hit_id', keep='last')
                indexes_to_drop = to_drop.index
                df_hits = df.drop(indexes_to_drop)

                output_file = os.path.join(DATA_PROCESSED_HITS, file_name_pkl)
                df_hits.to_pickle(output_file)

                file_flags["transformed"] = True
                file_flags["saved"] = True
                save_flags_hits(flags)

                hits_transform_success.inc()
                log_operation('INFO', f"Transformed data saved to file: {output_file}")

            except Exception as e:
                hits_transform_fail.inc()
                log_operation('ERROR', f"Error during transformation: {e}")
                raise
        else:
            log_operation('WARNING', f"Skipping transformation for {file_name}")

    log_operation('INFO', "All files have been processed.")

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10), retry=retry_if_exception_type(psycopg2.OperationalError))
def send_hits_to_db():
    if not os.path.exists(DATA_PROCESSED_HITS):
        log_operation('ERROR', f"Directory {DATA_PROCESSED_HITS} does not exist.")
        return

    flags = load_flags_hits()

    engine = create_engine(
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
        f"/{os.getenv('DB_NAME')}"
    )

    for file_name in os.listdir(DATA_PROCESSED_HITS):
        file_path = os.path.join(DATA_PROCESSED_HITS, file_name)

        if file_name not in flags["hits_data"]:
            continue

        file_flags = flags["hits_data"][file_name]
        if not file_flags.get("transformed") or file_flags.get("to_db"):
            continue

        try:
            log_operation('INFO', f"Sending data for {file_name} to database...")
            with open(file_path, 'rb') as file:
                df_hits = pd.DataFrame(pickle.load(file))

            try:
                df_hits.to_sql(
                    "hits",
                    con=engine,
                    if_exists="append",
                    index=False,
                    method='multi')

                flags["hits_data"][file_name]["to_db"] = True
                save_flags_hits(flags)
                hits_db_success.inc()
                log_operation('INFO', f"Data for {file_name} successfully saved to database.")

            except Exception as e:
                if "duplicate key value" in str(e).lower():
                    log_operation('WARNING',
                                  f"Skipped {file_name} (duplicates found in DB). "
                                  f"Marking as processed."
                                  )
                    flags['sessions_data'][file_name]['to_db'] = True
                    save_flags_hits(flags)
                    hits_db_fail.inc()
                else:
                    raise

        except Exception as e:
            hits_db_fail.inc()
            log_operation('ERROR',
                          f"Failed to process {file_name}: {type(e).__name__} - {str(e)}"
                          )
            continue

if __name__ == "__main__":
    try:
        log_operation('INFO', "ETL process started", es_required=True)

        parser = argparse.ArgumentParser(description="Transform and send to db")
        parser.add_argument("--process", action="store_true", help="Transforming")
        parser.add_argument("--to_db", action="store_true", help="Sending to db")
        args = parser.parse_args()

        if args.process:
            transform_new_files_hits()
        if args.to_db:
            send_hits_to_db()

        push_to_gateway('localhost:9091', job='hits_processing_job', registry=registry)
        log_operation('INFO', "ETL process completed successfully", es_required=True)

    except Exception as e:
        log_operation('ERROR',
                      f"ETL process crashed: {str(e)}",
                      es_required=True)
        raise