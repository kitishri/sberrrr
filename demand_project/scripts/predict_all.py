import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    r2_score, mean_absolute_percentage_error, max_error
)

load_dotenv()

def load_model_and_scalers(model_dir):
    """Загрузка модели и скейлеров"""
    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "model.keras")

    scaler_X = joblib.load(model_dir / "scalers" / "scaler_X.pkl")
    scaler_y = joblib.load(model_dir / "scalers" / "scaler_y.pkl")

    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)

    return model, scaler_X, scaler_y, config


def create_sequences(df, features, target, window_size):
    """Создание срезов для LSTM"""
    X_vals = df[features].values
    y_vals = df[target].values

    X_seq = []
    y_seq = []
    indices = []

    for i in range(window_size, len(df)):
        X_seq.append(X_vals[i - window_size:i])
        y_seq.append(y_vals[i])
        indices.append(df.index[i])

    return np.array(X_seq), np.array(y_seq), indices


def predict_lstm(model_dir, data_path, output_path):
    # 1. Загружаем модель и конфиг
    model, scaler_X, scaler_y, config = load_model_and_scalers(model_dir)

    # 2. Загружаем данные
    df = pd.read_parquet(data_path)

    # 3. Масштабируем
    df_scaled = df.copy()
    df_scaled[config['features']] = scaler_X.transform(df_scaled[config['features']])
    df_scaled[config['target']] = scaler_y.transform(df_scaled[[config['target']]])

    # 4. Формируем последовательности
    X_seq, y_seq, idx_seq = create_sequences(df_scaled, config['features'], config['target'], config['window_size'])

    # 5. Прогнозируем
    y_pred_scaled = model.predict(X_seq)
    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_pred = np.expm1(y_pred_log).flatten()

    y_true_log = scaler_y.inverse_transform(y_seq.reshape(-1, 1))
    y_true = np.expm1(y_true_log).flatten()

    # 6. Метрики
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    maxerr = max_error(y_true, y_pred)

    metrics_dict = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "R2": r2,
        "MAPE (%)": mape,
        "MaxError": maxerr
    }

    # 7. Сохраняем прогноз
    predictions_df = pd.DataFrame({
        "index": idx_seq,
        "true": y_true,
        "prediction": y_pred
    })
    output_path = Path(output_path)
    predictions_df.to_csv(output_path, index=False)

    # 8. Сохраняем метрики в JSON рядом с CSV
    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 9. Вывод в консоль
    print(f"Прогноз сохранён в {output_path}")
    print(f"Метрики сохранены в {metrics_path}")
    print("\n=== Метрики модели ===")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":

    HOST_CODE_PATH = Path(os.getenv("HOST_CODE_PATH"))
    DATA_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'final_data.parquet'
    MODEL_PATH = HOST_CODE_PATH / 'data' / 'models' / 'best_lstm_model'
    PREDICTIONS_PATH = HOST_CODE_PATH / 'data' / 'predictions' / 'predictions_all_data.csv'

    predict_lstm(MODEL_PATH, DATA_PATH, PREDICTIONS_PATH)
