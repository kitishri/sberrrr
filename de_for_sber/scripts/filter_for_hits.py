import pandas as pd

# Загрузка данных
hits = pd.read_pickle("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/processed_hits.pkl")
invalid_session_ids = pd.read_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/missing_session_ids.csv")

invalid_session_ids = invalid_session_ids['session_id'].tolist()

# Находим индексы строк, где session_id совпадает с invalid_session_ids
indexes_to_drop = hits[hits['session_id'].isin(invalid_session_ids)].index

# Удаляем эти строки
hits_filtered = hits.drop(index=indexes_to_drop)
hits_filtered.to_csv("C:/Users/Ekaterina/sber_de/de_for_sber/data/processed/df_sessions_filtered.csv", index=False)



