import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta
from configs.logging import log_step, logger


class AdvancedDemandForecastPipeline:
    def __init__(self, df, output_dir):
        self.df = df.copy()
        self.df_filtered = None
        self.output_dir = output_dir
        self.df_final = None

    @log_step('Preprocessing data')
    def preprocess(self):
        df = self.df.copy()

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

        return self

    @log_step('Extracting temporal features')
    def extract_temporal_features(self):
        dt = self.df_filtered['InvoiceDate'].dt
        temporal_features = {
            'Year': dt.year,
            'Month': dt.month,
            'Week': dt.isocalendar().week.astype(int),
            'Day': dt.day,
            'Weekday': dt.weekday + 1,
            'IsWeekend': ((dt.weekday + 1).isin([6, 7])).astype(int),  # 6=Saturday, 7=Sunday
            'HolidaySeason': dt.month.isin([11, 12]).astype(int),
            'IsStartOfMonth': dt.is_month_start.astype(int),
            'IsEndOfMonth': dt.is_month_end.astype(int),
            'BlackFriday': (dt.month == 11) & (dt.day >= 23) & (dt.day <= 29) & (dt.weekday == 4)
        }

        self.df_filtered = self.df_filtered.assign(**temporal_features)
        return self

    @log_step('Calculating RFM')
    def calculate_and_merge_rfm(self):
        df = self.df_filtered.dropna(subset=['CustomerID'])

        last_date = df['InvoiceDate'].max() + timedelta(days=1)

        rfm = df.groupby('CustomerID').agg(
            Recency=('InvoiceDate', lambda x: (last_date - x.max()).days),
            Frequency=('InvoiceNo', 'nunique'),
            Monetary=('TotalSum', 'sum')
        ).reset_index()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['RFM_Cluster'] = kmeans.fit_predict(rfm_scaled)

        self.df_filtered = self.df_filtered.merge(
            rfm[['CustomerID', 'RFM_Cluster']],
            on='CustomerID',
            how='left'
        )

        return self

    @log_step('Generating Lags and Rolling')
    def generate_lags_and_rolling(self):
        required_cols = {'StockCode', 'Quantity', 'InvoiceDate'}
        if not required_cols.issubset(self.df_filtered.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        clean_df = self.df_filtered[
            (self.df_filtered['Quantity'].between(1, 999)) &
            (self.df_filtered['UnitPrice'].between(0.01, 9999))
            ].copy()

        agg = (clean_df
               .groupby(["InvoiceDate", "StockCode"], as_index=False)
               ["Quantity"]
               .sum()
               .rename(columns={"Quantity": "Quantity_Agg"})
               .sort_values(["StockCode", "InvoiceDate"])
               .reset_index(drop=True))

        q_low, q_high = agg["Quantity_Agg"].quantile([0.01, 0.99])


        for lag in [1, 7, 30]:
            lag_vals = agg.groupby('StockCode')['Quantity_Agg'].shift(lag)
            agg[f'Lag_{lag}'] = lag_vals.clip(lower=q_low, upper=q_high).fillna(0).values

        for window in [7, 30]:
            group = agg.groupby('StockCode')['Quantity_Agg']

            rolling_mean = group.rolling(window=window, min_periods=1).mean()
            rolling_std = group.rolling(window=window, min_periods=1).std().fillna(1)
            agg[f'RollingNormMean_{window}'] = (
                (rolling_mean / rolling_std)
                .clip(lower=q_low, upper=q_high)
                .values
            )

            rolling_range = group.rolling(window=window, min_periods=1).max() - \
                            group.rolling(window=window, min_periods=1).min()
            agg[f'RollingRange_{window}'] = rolling_range.clip(lower=0, upper=q_high).fillna(0).values

        for name, (num, denom) in {'PctChange_7': ('Lag_1', 'Lag_7'),
                                   'PctChange_30': ('Lag_1', 'Lag_30')}.items():
            if denom in agg.columns:
                denom_vals = agg[denom].replace(0, np.nan)
                pct = (agg[num] - agg[denom]) / denom_vals
                agg[name] = pct.fillna(0).clip(-1, 1).values

        merge_cols = [c for c in agg.columns if c != 'Quantity_Agg']
        self.df_filtered = pd.merge(
            clean_df.reset_index(drop=True),
            agg[merge_cols].reset_index(drop=True),
            on=['InvoiceDate', 'StockCode'],
            how='left'
        )

        return self

    @log_step('Final steps')
    def filter_and_finalize(self):

        target = "log_Quantity"
        cols_to_drop = [
            'InvoiceNo', 'StockCode', 'Description', 'Country',
            'Quantity', 'TotalSum', 'UnitPrice'
        ]
        self.df_final = self.df_filtered.drop(
            columns=[col for col in cols_to_drop if col in self.df_filtered.columns])

        self.df_final = self.df_final.dropna()
        if self.df_final.empty:
            raise ValueError("Dataset is empty after NA removal")

        cat = ['IsWeekend', 'HolidaySeason', 'IsStartOfMonth', 'IsEndOfMonth', 'RFM_Cluster']

        numeric_cols = [
            col for col in self.df_final.select_dtypes(include=np.number).columns
            if col != target and col not in cat
        ]

        corr_matrix = self.df_final[numeric_cols + [target]].corr()
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        print(self.df_final.describe().T)

        self.selected_features = target_corr.index[1:16].tolist() + cat
        final_cols = self.selected_features + [target]

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

    def run_pipeline(self):
        ((self.preprocess()
         .extract_temporal_features()
         .calculate_and_merge_rfm()
         .generate_lags_and_rolling()
         .filter_and_finalize())
         .save_outputs())
        return self
