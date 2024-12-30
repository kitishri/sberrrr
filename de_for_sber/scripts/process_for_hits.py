import pandas as pd
import datetime

def process_hits():
    # Загружаем данные
    df_hits = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/raw/ga_hits.csv")

    # Обработка пропусков в данных hits
    df_hits['hit_referer'] = df_hits['hit_referer'].fillna('unknown')
    df_hits['event_label'] = df_hits['event_label'].fillna('unknown')
    df_hits['event_value'] = df_hits['event_value'].fillna(0)

    # Преобразование дат
    df_hits['hit_date'] = pd.to_datetime(df_hits['hit_date'], errors='coerce')

    # Удаление атрибута из-за большого количества NaN
    df_hits = df_hits.drop(columns='hit_time')

    # Создаем новый столбец 'hit_id', который является комбинацией 'session_id' и 'hit_number'
    df_hits['hit_id'] = df_hits['session_id'].astype(str) + '_' + df_hits['hit_number'].astype(str)

    # Шаг 1: Найдем все строки с одинаковыми hit_id (дубликаты)
    # Мы ищем строки, где 'hit_id' повторяется
    duplicates = df_hits[df_hits.duplicated(subset='hit_id', keep=False)]

    # Шаг 2: Для каждой группы одинаковых hit_id оставляем только первую строку

    df_hits = df_hits.drop_duplicates(subset='hit_id', keep='first')

    # Сохранение обработанных данных в новый CSV-файл
    df_hits.to_pickle("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/processed_hits.pkl")

    print("Hits processed and saved.")

def main():
        process_hits()  # Обрабатываем данные hits

if __name__ == "__main__":
    main()

