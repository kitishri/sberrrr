import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

path = '/opt/airflow'
os.environ['PROJECT_PATH'] = path
print("Текущая рабочая директория:", os.getcwd())
sys.path.insert(0, path)

from scripts.new_data_hits import (
    load_flags_hits, transform_new_files_hits, send_hits_to_db
)
from scripts.new_data_sessions import (
    load_flags_sessions, transform_new_files_sessions, send_sessions_to_db
)
from configs.logging_config import log_message

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='sber_dag',
    default_args=default_args,
    description='DAG for transforming and sending sessions/hits to DB',
    schedule_interval=None,
    catchup=False,
)

task_load_logger = PythonOperator(
    task_id='load_logger',
    python_callable=log_message,
    dag=dag,
)

# Sessions tasks
task_sessions_load_flags = PythonOperator(
    task_id='load_flags_sessions',
    python_callable=load_flags_sessions,
    dag=dag,
)
task_transform_sessions = PythonOperator(
    task_id='transform_sessions',
    python_callable=transform_new_files_sessions,
    dag=dag,
)

task_sessions_db = PythonOperator(
    task_id='send_sessions_to_db',
    python_callable=send_sessions_to_db,
    dag=dag,
)

# Hits tasks
task_hits_load_flags = PythonOperator(
    task_id='load_flags_hits',
    python_callable=load_flags_hits,
    dag=dag,
)

task_transform_hits = PythonOperator(
    task_id='transform_hits',
    python_callable=transform_new_files_hits,
    dag=dag,
)

task_hits_db = PythonOperator(
    task_id='send_hits_to_db',
    python_callable=send_hits_to_db,
    dag=dag,
)


task_load_logger >> [task_sessions_load_flags, task_hits_load_flags]

(task_sessions_load_flags >> task_transform_sessions >> task_sessions_db >>
 task_hits_load_flags >> task_transform_hits >> task_hits_db)





