from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LightGBMModel:
    def __init__(self, data_path, features_path, target, output_dir):
        self.data_path = Path(data_path)
        self.features_path = Path(features_path)
        self.target = target
        self.output_dir = Path(output_dir)
        self.df = self._load_data()
        self.features = self._load_features()

    def _load_data(self):
        return pd.read_parquet(self.data_path)

    def _load_features(self):
        import json
        with open(self.features_path, 'r') as f:
            return json.load(f)

    def _evaluate(self, y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2

    def cross_validate(self, n_splits=3):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []
        all_predictions = []

        best_rmse = float('inf')
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.df)):
            logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")

            df_train = self.df.iloc[train_idx]
            df_val = self.df.iloc[val_idx]

            X_train, y_train = df_train[self.features], df_train[self.target]
            X_val, y_val = df_val[self.features], df_val[self.target]

            model = LGBMRegressor(
                n_estimators=5000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(stopping_rounds=100),
                    log_evaluation(period=50)  # каждые 50 итераций будет лог
                ]
            )

            y_pred = model.predict(X_val)
            mse, rmse, mae, r2 = self._evaluate(y_val, y_pred)

            all_predictions.append(pd.DataFrame({
                "fold": fold + 1,
                "val_index": df_val.index,
                "true": y_val.values,
                "pred": y_pred
            }))

            logger.info(f"Fold {fold + 1} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = deepcopy(model)

            metrics.append({
                'fold': fold + 1,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

        if best_model:
            self.save_model(best_model, "best_lgbm_model")

        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = self.output_dir / "cv_predictions_lgbm.csv"
        all_predictions_df.to_csv(predictions_path, index=False)

        return metrics

    @staticmethod
    def plot_all(pred_csv: Path, df_parquet: Path, output_dir: Path, sample_size=5000):
        sns.set(style="whitegrid", font_scale=1.2)
        olive = "#708238"

        df_full = pd.read_parquet(df_parquet)
        df = pd.read_csv(pred_csv, low_memory=False).rename(columns={'true': 'y_true', 'pred': 'y_pred'})
        df['val_index'] = df['val_index'].astype(int)

        df_sample = (
            df.sample(sample_size, random_state=42).sort_values('val_index')
            if len(df) > sample_size else df.sort_values('val_index')
        )

        if {'val_index', 'y_true', 'y_pred'}.issubset(df_sample.columns):
            tail_df = df_sample.tail(500)
            plt.figure(figsize=(35, 8))
            plt.plot(tail_df['val_index'], tail_df['y_true'], label="Истина", color='green', linewidth=10, alpha=0.4)
            plt.plot(tail_df['val_index'], tail_df['y_pred'], label="Прогноз", color='brown', linewidth=2, alpha=1.0,
                     linestyle='--')
            plt.title("Истина vs Прогноз", fontsize=16)
            plt.xlabel("val_index")
            plt.ylabel("Значение")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'lgb' /  "true_vs_pred_sample.png", dpi=300)
            plt.close()

        folds = sorted(df['fold'].unique())
        r2_scores = [r2_score(df[df['fold'] == f]['y_true'], df[df['fold'] == f]['y_pred']) for f in folds]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=folds, y=r2_scores, palette=sns.light_palette(olive, n_colors=len(folds)))
        plt.title("R² по фолдам", fontsize=16)
        plt.xlabel("Fold")
        plt.ylabel("R²")
        plt.ylim(0, 5)
        for i, v in enumerate(r2_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'lgb' /  "r2_by_fold.png", dpi=300)
        plt.close()

        df_sample['delta_pct'] = (
                np.abs(df_sample['y_true'] - df_sample['y_pred']) /
                df_sample['y_true'].replace(0, np.nan) * 100
        ).clip(upper=100)

        rolling = df_sample['delta_pct'].rolling(window=200).mean()

        plt.figure(figsize=(20, 6))
        plt.plot(df_sample['val_index'], rolling, color='darkgreen', linewidth=2)
        plt.title("Скользящее среднее дельты (%)", fontsize=16)
        plt.xlabel("val_index")
        plt.ylabel("Отклонение, %")
        plt.ylim(0, 2)
        plt.tight_layout()
        plt.savefig(output_dir / 'lgb' /  "delta_pct.png", dpi=300)
        plt.close()

        merged = df_sample.merge(df_full, left_on='val_index', right_index=True, how='left')
        merged['InvoiceDate'] = pd.to_datetime(merged['InvoiceDate'])

        daily_cluster = merged.groupby(['InvoiceDate', 'RFM_Cluster'])['y_pred'].sum().reset_index()
        weekly = (
            daily_cluster
            .groupby([pd.Grouper(key='InvoiceDate', freq='W'), 'RFM_Cluster'])['y_pred']
            .sum()
            .reset_index()
        )

        top_clusters = (
            weekly.groupby('RFM_Cluster')['y_pred'].sum()
            .sort_values(ascending=False)
            .head(3)
            .index
        )
        weekly = weekly[weekly['RFM_Cluster'].isin(top_clusters)]
        palette = sns.color_palette("BuGn_d", len(top_clusters))  # <- другой зелёный стиль

        plt.figure(figsize=(20, 6))
        for i, (cluster, group) in enumerate(weekly.groupby('RFM_Cluster')):
            plt.plot(
                group['InvoiceDate'],
                group['y_pred'],
                label=f"Кластер {cluster}",
                color=palette[i],
                linestyle='-',
                linewidth=2,
                marker='o',
                markersize=5
            )
        plt.title("Прогноз по кластерам (недельная агрегация, топ-3)", fontsize=16)
        plt.xlabel("Дата")
        plt.ylabel("Спрос")
        plt.xticks(rotation=45)
        plt.legend(title='RFM Кластер')
        plt.tight_layout()
        plt.savefig(output_dir / 'lgb' /  "pred_by_cluster_time_sample.png", dpi=300)
        plt.close()

    def save_model(self, model, model_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or f"lgbm_model_{timestamp}"
        model_path = self.output_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path / "model.pkl")
        config = {
            'target': self.target,
            'features': self.features,
            'model_name': model_name,
            'saved_at': timestamp
        }
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Модель сохранена в {model_path}")
