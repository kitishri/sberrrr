import os
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from copy import deepcopy
from pathlib import Path
from configs.logging import log_step, logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class LSTMModel:
    def __init__(self, data_path, features_path, target, window_size, output_dir):

        self.data_path = Path(data_path)
        self.features_path = Path(features_path)
        self.target = target
        self.window_size = window_size
        self.output_dir = Path(output_dir)

        self.df = self._load_data()
        self.features = self._load_features()

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()


    def _load_data(self):

        df = pd.read_parquet(self.data_path)
        return df

    def _load_features(self):

        with open(self.features_path, 'r') as f:
            features = json.load(f)

        return features


    def _scale_and_write_back(self, df_train, df_test):
        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()

        df_train_scaled[self.features] = self.scaler_X.fit_transform(df_train[self.features])
        df_test_scaled[self.features] = self.scaler_X.transform(df_test[self.features])

        df_train_scaled[self.target] = self.scaler_y.fit_transform(df_train[[self.target]])
        df_test_scaled[self.target] = self.scaler_y.transform(df_test[[self.target]])

        return df_train_scaled, df_test_scaled

    def _create_sequences(self, df):
        X_vals = df[self.features].values
        y_vals = df[self.target].values
        n_samples = len(df) - self.window_size

        X_seq = np.zeros((n_samples, self.window_size, len(self.features)))
        y_seq = np.zeros(n_samples)

        for i in range(self.window_size, len(df)):
            X_seq[i - self.window_size] = X_vals[i - self.window_size:i]
            y_seq[i - self.window_size] = y_vals[i]

        indices = df.index[self.window_size:]

        return X_seq, y_seq, indices

    def _build_model(self):
        model = Sequential([
            LSTM(
                64,
                activation='tanh',
                input_shape=(self.window_size, len(self.features))
            ),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=Huber(), metrics=['mae'])
        return model

    def _evaluate(self, model, X_test_seq, y_test_seq):
        y_pred_scaled = model.predict(X_test_seq)

        y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_true_log = self.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))

        y_pred = np.expm1(y_pred_log).flatten()
        y_true = np.expm1(y_true_log).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        predictions_df = pd.DataFrame({
            "true": y_true,
            "pred": y_pred
        })

        return mse, mae, r2, predictions_df

    def cross_validate(self, n_splits=3, epochs=30, batch_size=64):
        logger.info(f"Начало кросс-валидации (n_splits={n_splits}, epochs={epochs})")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []
        all_predictions = []

        best_rmse = float('inf')
        best_model = None
        best_scaler_X = None
        best_scaler_y = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.df)):
            logger.info(f"\n--- Fold {fold + 1}/{n_splits} ---")
            logger.info(f"Train indices: {train_idx.min()} to {train_idx.max()}")
            logger.info(f"Val indices: {val_idx.min()} to {val_idx.max()}")

            df_train = self.df.iloc[train_idx].copy()
            df_val = self.df.iloc[val_idx].copy()

            df_train_scaled, df_val_scaled = self._scale_and_write_back(df_train, df_val)
            X_train, y_train, idx_train = self._create_sequences(df_train_scaled)
            X_val, y_val, idx_val = self._create_sequences(df_val_scaled)

            self.model = self._build_model()
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            logger.info("Обучение модели...")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )

            mse, mae, r2, predictions_df = self._evaluate(self.model, X_val, y_val)
            rmse = np.sqrt(mse)


            predictions_df["fold"] = fold + 1
            predictions_df["val_index"] = idx_val
            all_predictions.append(predictions_df)

            logger.info(f"Fold {fold + 1} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

            self.save_model(f"fold_{fold + 1}_model")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = clone_model(self.model)
                best_model.set_weights(self.model.get_weights())
                best_scaler_X = deepcopy(self.scaler_X)
                best_scaler_y = deepcopy(self.scaler_y)
                best_history = history.history

            metrics.append({
                'fold': fold + 1,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            if best_model and self.output_dir:
                self.model = best_model
                self.scaler_X = best_scaler_X
                self.scaler_y = best_scaler_y
                self.save_model("best_lstm_model")

        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = self.output_dir / "cv_predictions.csv"
        all_predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Предсказания по фолдам сохранены: {predictions_path}")

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
            plt.plot(tail_df['val_index'], tail_df['y_true'], label="Истина", color='green', linewidth=10, alpha=0.4,
                     marker='o',
                     markersize=4)
            plt.plot(tail_df['val_index'], tail_df['y_pred'], label="Прогноз", color='gray', linewidth=2,
                     linestyle='--', marker='x', markersize=4)
            plt.title("Истина vs Прогноз (последние 1000)")
            plt.xlabel("val_index")
            plt.ylabel("Значение")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'lstm' /  "true_vs_pred_sample.png", dpi=300)
            plt.close()

        folds = sorted(df['fold'].unique())
        r2_scores = [r2_score(df[df['fold'] == f]['y_true'], df[df['fold'] == f]['y_pred']) for f in folds]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=folds, y=r2_scores, palette=sns.light_palette(olive, n_colors=len(folds)))
        plt.title("R² по фолдам")
        plt.xlabel("Fold")
        plt.ylabel("R²")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm' / "r2_by_fold.png", dpi=300)
        plt.close()

        df_sample['delta_pct'] = (
                np.abs(df_sample['y_true'] - df_sample['y_pred']) /
                df_sample['y_true'].replace(0, np.nan) * 100
        ).clip(upper=100)

        rolling = df_sample['delta_pct'].rolling(window=50).mean()

        plt.figure(figsize=(16, 6))
        plt.plot(df_sample['val_index'], rolling, color='darkgreen', linewidth=2)
        plt.title("Скользящее среднее дельты (%)")
        plt.xlabel("val_index")
        plt.ylabel("Отклонение, %")
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm' / "delta_pct.png", dpi=300)
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
        palette = sns.color_palette("Greens_d", len(top_clusters))

        plt.figure(figsize=(16, 6))
        for i, (cluster, group) in enumerate(weekly.groupby('RFM_Cluster')):
            plt.plot(
                group['InvoiceDate'],
                group['y_pred'],
                label=f"Кластер {cluster}",
                color=palette[i],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                alpha=0.9
            )
        plt.title("Прогноз по кластерам (недельная агрегация, топ-3)")
        plt.xlabel("Дата")
        plt.ylabel("Спрос")
        plt.xticks(rotation=45)
        plt.legend(title='RFM Кластер')
        plt.tight_layout()
        plt.savefig(output_dir / 'lstm' / "pred_by_cluster_time_sample.png", dpi=300)
        plt.close()

    def save_model(self, model_name=None, history=None):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or f"lstm_model_{timestamp}"
        model_path = self.output_dir / model_name

        model_path.mkdir(parents=True, exist_ok=True)

        self.model.save(model_path / "model.keras")

        scaler_path = model_path / "scalers"
        scaler_path.mkdir(exist_ok=True)
        joblib.dump(self.scaler_X, scaler_path / "scaler_X.pkl")
        joblib.dump(self.scaler_y, scaler_path / "scaler_y.pkl")

        config = {
            'target': self.target,
            'window_size': self.window_size,
            'features': self.features,
            'model_name': model_name,
            'saved_at': timestamp
        }
        with open(model_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        if history:
            with open(model_path / "history.json", 'w') as f:
                json.dump(history, f, indent=2)

        logger.info(f"Модель и компоненты сохранены в {model_path}")
        return model_path




