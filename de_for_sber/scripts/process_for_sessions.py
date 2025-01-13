
import pandas as pd



def process_sessions():
    # Загружаем данные
    df_sessions = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_sessions.csv")

    # Обработка пропусков в данных sessions
    df_sessions['utm_campaign'] = df_sessions['utm_campaign'].fillna('unknown')
    df_sessions['utm_adcontent'] = df_sessions['utm_adcontent'].fillna('unknown')
    df_sessions['utm_keyword'] = df_sessions['utm_keyword'].fillna('unknown')
    df_sessions['device_os'] = df_sessions['device_os'].fillna('unknown')
    df_sessions['device_brand'] = df_sessions['device_brand'].fillna('unknown')
    df_sessions['utm_source'] = df_sessions['utm_source'].fillna('unknown')

    # Преобразование дат, заменить NaT на дефолтное значение
    df_sessions["visit_date"] = pd.to_datetime(df_sessions["visit_date"], errors='coerce')

    # Преобразование времени
    df_sessions['visit_time'] = pd.to_datetime(df_sessions['visit_time'], format='%H:%M:%S').dt.time
    # Преобразование типов данных
    df_sessions["client_id"] = df_sessions["client_id"].astype(str)

    # Удаление ненужной колонки
    df_sessions = df_sessions.drop(columns='device_model', axis=1)

    # Сохранение обработанных данных
    df_sessions.to_pickle("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/processed_sessions.pkl")
    print("Sessions processed and saved.")

def main():

    process_sessions()  # Обрабатываем данные sessions


if __name__ == "__main__":
    main()

