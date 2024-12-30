import pandas as pd
import datetime
pd.set_option('display.max_columns', None)


def process_hits():
    # Загружаем данные
    df_hits = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_hits.csv")

    # Обработка пропусков в данных hits
    df_hits['hit_referer'] = df_hits['hit_referer'].fillna('unknown')
    df_hits['event_label'] = df_hits['event_label'].fillna('unknown')
    df_hits['event_value'] = df_hits['event_value'].fillna(0)

    # Преобразование дат
    df_hits['hit_date'] = pd.to_datetime(df_hits['hit_date'], errors='coerce')

    # Преобразование времени
    def convert_milliseconds_to_time(ms):
        if pd.isna(ms) or ms == "":
            return None  # Для корректной записи в базу, если значение отсутствует
        total_seconds = int(ms) // 1000
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return datetime.time(hour=hours, minute=minutes, second=seconds)

    # Применяем функцию к столбцу 'hit_time'
    df_hits['hit_time'] = df_hits['hit_time'].apply(convert_milliseconds_to_time)





    # Создаем новый столбец 'hit_id', который является комбинацией 'session_id' и 'hit_number'
    df_hits['hit_id'] = df_hits['session_id'].astype(str) + '_' + df_hits['hit_number'].astype(str)

    # Шаг 1: Найдем все строки с одинаковыми hit_id (дубликаты)
    # Мы ищем строки, где 'hit_id' повторяется
    duplicates = df_hits[df_hits.duplicated(subset='hit_id', keep=False)]

    # Шаг 2: Для каждой группы одинаковых hit_id оставляем только первую строку
    # Это удалит все повторяющиеся строки, оставив только первую для каждого уникального hit_id
    df_hits = df_hits.drop_duplicates(subset='hit_id', keep='first')

    # Сохранение обработанных данных в новый CSV-файл
    df_hits.to_pickle("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/processed_hits.pkl")

    print("Hits processed and saved.")


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
    process_hits()  # Обрабатываем данные hits
    process_sessions()  # Обрабатываем данные sessions


if __name__ == "__main__":
    main()

