import pandas as pd



def new():
    def load_data():
        df_hits = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_hits.csv")
        df_sessions = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_sessions.csv", low_memory=False)
        return df_hits, df_sessions

    pd.set_option('display.max_columns', None)

    print(df_sessions.shape, df_hits.shape)

    def process_nan_hits(df_hits):
        df_hits = df_hits.copy()
        df_hits['hit_referer'] = df_hits['hit_referer'].fillna('unknown')
        df_hits['event_label'] = df_hits['event_label'].fillna('unknown')
        df_hits['event_value'] = df_hits['event_value'].fillna(0)
        return df_hits

    def drop_in_sessions(df_sessions):
        return df_sessions.drop(columns='device_model', axis=1)

    def process_nan_sessions(df_sessions):
        df_sessions = df_sessions.copy()
        df_sessions['utm_campaign'] = df_sessions['utm_campaign'].fillna('unknown')
        df_sessions['utm_adcontent'] = df_sessions['utm_adcontent'].fillna('unknown')
        df_sessions['utm_keyword'] = df_sessions['utm_keyword'].fillna('unknown')
        df_sessions['device_os'] = df_sessions['device_os'].fillna('unknown')
        df_sessions['device_model'] = df_sessions['device_model'].fillna('unknown')
        return df_sessions

    def transform_date(df_sessions, df_hits):
        df_sessions = df_sessions.copy()
        df_hits = df_hits.copy()
        df_sessions['visit_date'] = pd.to_datetime(df_sessions['visit_date'], format='%Y-%m-%d')
        df_hits['hit_date'] = pd.to_datetime(df_hits['hit_date'], format='%Y-%m-%d')
        return df_sessions, df_hits

    def transform_time_s(df_sessions):
        df_sessions = df_sessions.copy()
        df_sessions['visit_time'] = pd.to_datetime(df_sessions['visit_time'], format='%H:%M:%S')
        return df_sessions

    def transform_time_h(df_hits):
        def convert_milliseconds_to_time(ms):
            if pd.isna(ms):
                return "00:00:00"

            total_seconds = ms // 1000
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        df_hits = df_hits.copy()

        df_hits['hit_time'] = df_hits['hit_time'].apply(convert_milliseconds_to_time)

        df_hits['hit_time'] = pd.to_timedelta(df_hits['hit_time'])
        return df_hits

    def index_changes(df_hits, df_sessions):
        df_hits = df_hits.copy()
        df_sessions = df_sessions.copy()
        df_hits.set_index('session_id', inplace=True)
        df_sessions.set_index('session_id', inplace=True)
        return df_hits, df_sessions







if __name__ == "__main__":
    new()