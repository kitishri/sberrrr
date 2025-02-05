from new_data_hits import load_flags_hits, load_new_files_hits, transform_hits, send_hits_to_db
from new_data_sessions import load_new_files_sessions, load_flags_sessions, transform_sessions, send_sessions_to_db

def main():

    load_flags_sessions()
    dataframes_sessions, file_paths_s = load_new_files_sessions()
    transform_sessions(dataframes_sessions, file_paths_s)
    send_sessions_to_db()

    load_flags_hits()
    dataframes_hits, file_paths = load_new_files_hits()
    transform_hits(dataframes_hits, file_paths)
    send_hits_to_db()

if __name__ == "__main__":
    main()
