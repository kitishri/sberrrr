import numpy as np
import json
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta
from configs.logging import log_step
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedDemandForecastPipeline:
    def __init__(self, df, output_dir):
        self.df = df.copy()
        self.df_filtered = None
        self.output_dir = output_dir
        self.df_final = None

    @log_step('Preprocessing data')
    def preprocess(self):
        df = self.df.copy()
        print(df.shape)


        df = df[df['UnitPrice'] > 0].copy()

        df = df.drop_duplicates(subset=df.columns.difference(['InvoiceNo']))

        df['log_Quantity'] = np.log1p(df['Quantity'].clip(lower=0))
        df['log_UnitPrice'] = np.log1p(df['UnitPrice'].clip(lower=1e-6))
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']
        df['log_TotalSum'] = np.log1p((df['Quantity'].clip(lower=0) * df['UnitPrice'].clip(lower=1e-6)).clip(lower=0))
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TransactionHour'] = df['InvoiceDate'].dt.hour
        df['DayOfYear'] = df['InvoiceDate'].dt.dayofyear

        self.df_filtered = df

        print(pd.__version__)
        print(np.__version__)
        print(sklearn.__version__)

        return self

    @log_step('Extracting temporal features')
    def extract_temporal_features(self):
        dt = self.df_filtered['InvoiceDate'].dt
        temporal_features = {
            'Year': dt.year,
            'Month': dt.month,
            'Weekday': dt.weekday + 1,
            'IsWeekend': ((dt.weekday + 1).isin([6, 7])).astype(int),
            'HolidaySeason': dt.month.isin([11, 12]).astype(int),
            'IsStartOfMonth': dt.is_month_start.astype(int),
            'IsEndOfMonth': dt.is_month_end.astype(int),
            'BlackFriday': (dt.month == 11) & (dt.day >= 23) & (dt.day <= 29) & (dt.weekday == 4)
        }

        self.df_filtered = self.df_filtered.assign(**temporal_features)

        self.df_filtered['Month_sin'] = np.sin(2 * np.pi * self.df_filtered['Month'] / 12)
        self.df_filtered['Month_cos'] = np.cos(2 * np.pi * self.df_filtered['Month'] / 12)
        self.df_filtered['Weekday_sin'] = np.sin(2 * np.pi * self.df_filtered['Weekday'] / 7)
        self.df_filtered['Weekday_cos'] = np.cos(2 * np.pi * self.df_filtered['Weekday'] / 7)
        top_10_items = self.df_filtered['StockCode'].value_counts().head(10).index
        self.df_filtered['IsPopularItem'] = self.df_filtered['StockCode'].isin(top_10_items).astype(int)
        unitprice_threshold = self.df_filtered.groupby('StockCode')['UnitPrice'].transform('median')
        self.df_filtered['IsDiscounted'] = (self.df_filtered['UnitPrice'] < unitprice_threshold).astype(int)
        print(self.df_filtered.describe().T)


        return self

    @log_step('Calculating RFM')
    def calculate_and_merge_rfm(self):

        df = self.df_filtered.dropna(subset=['CustomerID'])
        df_filtered = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

        q_low_qty = df_filtered['Quantity'].quantile(0.01)
        q_high_qty = df_filtered['Quantity'].quantile(0.99)
        q_low_price = df_filtered['UnitPrice'].quantile(0.01)
        q_high_price = df_filtered['UnitPrice'].quantile(0.99)

        df_filtered = df_filtered[
            (df_filtered['Quantity'] >= q_low_qty) & (df_filtered['Quantity'] <= q_high_qty) &
            (df_filtered['UnitPrice'] >= q_low_price) & (df_filtered['UnitPrice'] <= q_high_price)
            ]

        df_filtered['TotalSum'] = df_filtered['Quantity'] * df_filtered['UnitPrice']

        last_date = df_filtered['InvoiceDate'].max() + timedelta(days=1)

        rfm = df_filtered.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (last_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalSum', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['RFM_Cluster'] = kmeans.fit_predict(rfm_scaled)
        cluster_stats = rfm.groupby('RFM_Cluster').agg(
            Customers=('CustomerID', 'count'),
            Recency_Mean=('Recency', 'mean'),
            Frequency_Mean=('Frequency', 'mean'),
            Monetary_Mean=('Monetary', 'mean')
        ).reset_index()

        for idx, row in cluster_stats.iterrows():
            print(f"Кластер {int(row['RFM_Cluster'])}:")
            print(f"  - Клиентов: {row['Customers']:,}")
            print(f"  - Среднее Recency: {row['Recency_Mean']:.2f}")
            print(f"  - Среднее Frequency: {row['Frequency_Mean']:.2f}")
            print(f"  - Среднее Monetary: {row['Monetary_Mean']:.2f}")
            print()

        self.df_filtered = self.df_filtered.merge(
            rfm[['CustomerID', 'RFM_Cluster']],
            on='CustomerID',
            how='left'
        )

        return self

    @log_step('Generating Lags and Rolling')
    def generate_lags_and_rolling(self):
        required_cols = {'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice'}
        if not required_cols.issubset(self.df_filtered.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Создаем агрегированный датафрейм с фильтрацией по разумным значениям
        agg_df = (
            self.df_filtered[
                (self.df_filtered['Quantity'] > 0) &
                (self.df_filtered['UnitPrice'] > 0)
                ]
            .groupby(['InvoiceDate', 'StockCode'], as_index=False)['Quantity']
            .sum()
            .rename(columns={'Quantity': 'Quantity_Agg'})
            .sort_values(['StockCode', 'InvoiceDate'])
            .reset_index(drop=True)
        )

        # Квантили для обрезки лагов
        q_low_lag, q_high_lag = agg_df['Quantity_Agg'].quantile([0.01, 0.99])

        # Создаем лаги с обрезкой и сбросом индексов
        for lag in [1, 7, 14, 30, 60]:
            lag_col = f'Lag_{lag}'
            lag_vals = agg_df.groupby('StockCode')['Quantity_Agg'].shift(lag)
            agg_df[lag_col] = lag_vals.clip(lower=q_low_lag, upper=q_high_lag).fillna(0).reset_index(level=0, drop=True)

        # Скользящие признаки
        for window in [7, 14, 30, 60]:
            group = agg_df.groupby('StockCode')['Quantity_Agg']

            rolling_mean = group.rolling(window=window, min_periods=1).mean()
            rolling_std = group.rolling(window=window, min_periods=1).std().fillna(1)
            rolling_std_safe = rolling_std.replace(0, 1e-9)
            rolling_norm_mean = rolling_mean / rolling_std_safe

            q_low_rnm, q_high_rnm = pd.Series(rolling_norm_mean.values).quantile([0.01, 0.99])
            col_norm = f'RollingNormMean_{window}'
            agg_df[col_norm] = rolling_norm_mean.clip(lower=q_low_rnm, upper=q_high_rnm).reset_index(level=0, drop=True)

            rolling_range = group.rolling(window=window, min_periods=1).max() - group.rolling(window=window,
                                                                                              min_periods=1).min()
            q_low_rr, q_high_rr = rolling_range.quantile([0.01, 0.99])
            col_range = f'RollingRange_{window}'
            agg_df[col_range] = rolling_range.clip(lower=q_low_rr, upper=q_high_rr).fillna(0).reset_index(level=0,
                                                                                                          drop=True)

            pct_change = group.pct_change(periods=window).fillna(0)
            col_pct = f'RollingPctChange_{window}'
            agg_df[col_pct] = pct_change.clip(-1, 1).reset_index(level=0, drop=True)

        # Относительные изменения между лагами
        for name, (num_col, denom_col) in {
            'PctChange_7': ('Lag_1', 'Lag_7'),
            'PctChange_14': ('Lag_1', 'Lag_14'),
            'PctChange_30': ('Lag_1', 'Lag_30'),
            'PctChange_60': ('Lag_1', 'Lag_60')
        }.items():
            if num_col in agg_df.columns and denom_col in agg_df.columns:
                denom_vals = agg_df[denom_col].replace(0, np.nan)
                pct = (agg_df[num_col] - agg_df[denom_col]) / denom_vals
                agg_df[name] = pct.fillna(0).clip(-1, 1).reset_index(level=0, drop=True)

        # Выбираем колонки с новыми признаками
        merge_cols = [c for c in agg_df.columns if c != 'Quantity_Agg']

        # Объединяем агрегированные фичи с оригинальным датафреймом без изменения исходных колонок
        self.df_filtered = self.df_filtered.merge(
            agg_df[merge_cols],
            on=['InvoiceDate', 'StockCode'],
            how='left'
        )

        return self

    @log_step('Final steps')
    def filter_and_finalize(self):


        cols_to_drop = ['Quantity', 'TotalSum', 'UnitPrice'
        ]
        self.df_final = self.df_filtered.drop(
            columns=[col for col in cols_to_drop if col in self.df_filtered.columns])

        self.df_final = self.df_final.dropna()
        if self.df_final.empty:
            raise ValueError("Dataset is empty after NA removal")
        print(self.df_final.describe().T)

        target = "log_Quantity"


        cat = ['HolidaySeason', 'RFM_Cluster', 'IsPopularItem', 'IsDiscounted'
        ]

        numeric_cols = [
            col for col in self.df_final.select_dtypes(include=np.number).columns
            if col != target and col not in cat
        ]

        corr_matrix = self.df_final[numeric_cols + [target]].corr()
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        print(target_corr)

        top_features = target_corr.drop(target).head(15).index.tolist()

        self.selected_features = top_features + cat



        if 'InvoiceDate' in self.df_final.columns:
            self.df_final = self.df_final.sort_values('InvoiceDate').reset_index(drop=True)

        return self

    def save_outputs(self):
        files = {
            'final_data': self.output_dir / 'final_data.parquet',
            'good_features': self.output_dir / 'good_features.json'
        }

        self.df_final.to_parquet(files['final_data'], index=False)

        files['good_features'].write_text(json.dumps(self.selected_features))

        return files

    def plot_raw_data(self):
        output_dir = self.output_dir / "plots_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        df = self.df_filtered.copy()

        sns.set(style="whitegrid")
        olive = "#708238"
        olive_palette = sns.light_palette(olive, n_colors=10, reverse=False)

        plt.figure(figsize=(8, 5))
        sns.histplot(df['Quantity'], bins=50, kde=True, color=olive)
        plt.title("Распределение таргета Quantity (сырые данные)")
        plt.xlabel("Quantity")
        plt.ylabel("Частота")
        plt.tight_layout()
        plt.savefig(output_dir / "target_distribution_raw.png")
        plt.close()

        q_low, q_high = df['Quantity'].quantile([0.01, 0.99])
        df_trimmed = df[(df['Quantity'] >= q_low) & (df['Quantity'] <= q_high)].copy()
        df_trimmed = df_trimmed[df_trimmed['Quantity'] > 0].copy()
        df_trimmed['log_quantity'] = np.log1p(df_trimmed['Quantity'])

        plt.figure(figsize=(8, 5))
        sns.histplot(df_trimmed['log_quantity'], bins=50, kde=True, color=olive)
        plt.title("Распределение log(Quantity) (1-99 квантиль, >0)")
        plt.xlabel("log(Quantity + 1)")
        plt.ylabel("Частота")
        plt.tight_layout()
        plt.savefig(output_dir / "target_distribution_log_trimmed.png")
        plt.close()

        top10 = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        top10_df = top10.reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top10_df,
            x='Quantity',
            y='Description',
            hue='Description',
            palette=olive_palette,
            legend=False
        )
        plt.title("Топ 10 товаров по продажам")
        plt.xlabel("Суммарное количество")
        plt.ylabel("Товар")
        plt.tight_layout()
        plt.savefig(output_dir / "top10_products.png")
        plt.close()

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['month'] = df['InvoiceDate'].dt.to_period('M')
        monthly = df.groupby('month')['Quantity'].mean()

        plt.figure(figsize=(10, 5))
        monthly.plot(marker='o', color=olive)
        plt.title("Средний спрос по месяцам")
        plt.xlabel("Месяц")
        plt.ylabel("Среднее количество")
        plt.tight_layout()
        plt.savefig(output_dir / "monthly_avg_demand.png")
        plt.close()

    def run_pipeline(self):
        ((self.preprocess()
         .extract_temporal_features()
         .calculate_and_merge_rfm()
         .generate_lags_and_rolling()
         .filter_and_finalize())
         .save_outputs())
        return self



