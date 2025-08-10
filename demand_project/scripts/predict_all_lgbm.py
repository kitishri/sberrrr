import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    r2_score, mean_absolute_percentage_error, max_error
)
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

def load_lgbm_model(model_dir):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")

    with open(model_dir / "config.json", 'r') as f:
        config = json.load(f)

    return model, config


def predict_lgbm(model_dir, data_path, output_path):
    model, config = load_lgbm_model(model_dir)

    df = pd.read_parquet(data_path)

    X = df[config['features']]
    y_true = df[config['target']]

    y_pred = model.predict(X)
    y_true = np.expm1(y_true)
    y_pred = np.expm1(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
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

    predictions_df = pd.DataFrame({
        "val_index": df.index,
        "true": y_true,
        "prediction": y_pred
    })
    output_path = Path(output_path)
    predictions_df.to_csv(output_path, index=False)

    metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")

def analyze_segments_during_spikes(df_full: Path, pred_path: Path, output_dir: Path):
    df_full = pd.read_parquet(df_full)
    df_pred = pd.read_csv(pred_path, low_memory=False).rename(columns={'true': 'y_true', 'prediction': 'y_pred'})
    df_pred['val_index'] = df_pred['val_index'].astype(int)
    df_pred = df_pred.copy()
    print(df_full['RFM_Cluster'].value_counts(normalize=True))

    pass

    df_pred['InvoiceDate'] = df_full.loc[df_pred['val_index'], 'InvoiceDate'].values
    df_pred['CustomerID'] = df_full.loc[df_pred['val_index'], 'CustomerID'].values
    df_pred['StockCode'] = df_full.loc[df_pred['val_index'], 'StockCode'].values

    df_pred['InvoiceDate'] = pd.to_datetime(df_pred['InvoiceDate'])

    cutoff_date = df_pred['InvoiceDate'].max() - pd.Timedelta(days=60)
    recent_pred = df_pred[df_pred['InvoiceDate'] >= cutoff_date]

    threshold = recent_pred['y_pred'].quantile(0.99)
    spike_days = recent_pred[recent_pred['y_pred'] > threshold]['InvoiceDate'].unique()

    spike_clients = df_pred[df_pred['InvoiceDate'].isin(spike_days)][['CustomerID']].dropna()

    print("spike_clients shape:", spike_clients.shape)
    print("df_full unique CustomerID:", df_full['CustomerID'].nunique())
    df_spike_segments = spike_clients.merge(
        df_full[['CustomerID', 'RFM_Cluster']].drop_duplicates('CustomerID'),
        on='CustomerID',
        how='left'
    )

    segment_counts = (
        df_spike_segments.groupby('RFM_Cluster')['CustomerID']
        .nunique()
        .reset_index()
        .rename(columns={'CustomerID': 'NumClients'})
        .sort_values('NumClients', ascending=False)
    )
    segment_counts['Share'] = segment_counts['NumClients'] / segment_counts['NumClients'].sum()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=segment_counts, x='RFM_Cluster', y='Share', palette='Greens')
    plt.title('Доля сегментов среди покупателей во время всплесков')
    plt.ylabel('Доля клиентов')
    plt.xlabel('RFM-сегмент')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "spike_segment_distribution.png", dpi=300)
    plt.close()

    all_clients_segments = df_full[['CustomerID', 'RFM_Cluster']].dropna().drop_duplicates()
    all_segment_counts = (
        all_clients_segments.groupby('RFM_Cluster')['CustomerID']
        .nunique()
        .reset_index()
        .rename(columns={'CustomerID': 'TotalClients'})
    )
    all_segment_counts['TotalShare'] = all_segment_counts['TotalClients'] / all_segment_counts['TotalClients'].sum()

    comparison = segment_counts.merge(all_segment_counts, on='RFM_Cluster', how='left')
    comparison['Lift'] = comparison['Share'] / comparison['TotalShare']

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x='RFM_Cluster', y='Lift', palette='coolwarm')
    plt.title('LIFT: Насколько сегмент чаще обычного покупает в дни всплесков')
    plt.ylabel('Кратность превышения')
    plt.xlabel('RFM-сегмент')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_lift_comparison.png", dpi=300)
    plt.close()

    return segment_counts

if __name__ == "__main__":

    HOST_CODE_PATH = Path(os.getenv("HOST_CODE_PATH"))
    DATA_PATH = HOST_CODE_PATH / 'data' / 'processed' / 'final_data.parquet'
    MODEL_PATH = HOST_CODE_PATH / 'data' / 'models' / 'best_lgbm_model'
    PREDICTIONS_PATH = HOST_CODE_PATH / 'data' / 'predictions' / 'predictions_all_data_lgbm.csv'
    OUTPUT_DIR_FIGURES = HOST_CODE_PATH / 'reports' / 'figures'

    predict_lgbm(MODEL_PATH, DATA_PATH, PREDICTIONS_PATH)
    analyze_segments_during_spikes(df_full=DATA_PATH, pred_path=PREDICTIONS_PATH, output_dir=OUTPUT_DIR_FIGURES)
