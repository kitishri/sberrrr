from new_data_hits import load_flags_hits, transform_new_files_hits, send_hits_to_db
from new_data_sessions import load_flags_sessions, transform_new_files_sessions, send_sessions_to_db

def main():

    load_flags_sessions()
    transform_new_files_sessions()
    send_sessions_to_db()

    load_flags_hits()
    transform_new_files_hits()
    send_hits_to_db()

if __name__ == "__main__":
    main()
