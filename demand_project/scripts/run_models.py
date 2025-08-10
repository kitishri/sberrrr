import os
from pathlib import Path
from dotenv import load_dotenv
from model_lstm import LSTMModel
from model_lgb import LightGBMModel

load_dotenv()

HOST_CODE_PATH = Path(os.getenv("HOST_CODE_PATH"))
CONTAINER_CODE_PATH = Path(os.getenv("CONTAINER_CODE_PATH"))

DATA_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'final_data.parquet'
FEATURES_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'good_features.json'

OUTPUT_DIR = HOST_CODE_PATH / 'data' / 'models'
OUTPUT_DIR_FIGURES = HOST_CODE_PATH / 'reports' / 'figures'
PREDICTIONS_PATH = HOST_CODE_PATH / 'data' / 'predictions'

READY_PRED_LSTM = HOST_CODE_PATH / 'data' / 'models' / 'cv_predictions.csv'
READY_PRED_LGB = HOST_CODE_PATH / 'data' / 'models' / 'cv_predictions_lgbm.csv'

TARGET = "log_Quantity"

model = LSTMModel(data_path=DATA_PATH, window_size=7, features_path=FEATURES_PATH, target=TARGET, output_dir=OUTPUT_DIR)
#cv_metrics = model.cross_validate()
model.plot_all(
    pred_csv=READY_PRED_LSTM,
    df_parquet=DATA_PATH,
    output_dir=OUTPUT_DIR_FIGURES,
    sample_size=5000)

"""
model_lgb = LightGBMModel(data_path=DATA_PATH, features_path=FEATURES_PATH, target=TARGET, output_dir=OUTPUT_DIR)
cv_metrics_lgbm = model_lgb.cross_validate()
model_lgb.plot_all(
    pred_csv=READY_PRED_LGB,
    df_parquet=DATA_PATH,
    output_dir=OUTPUT_DIR_FIGURES,
    sample_size=5000)
"""

