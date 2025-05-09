import json
from unittest.mock import patch, MagicMock

@patch("scripts.new_data_hits.get_redis_connection")
@patch("scripts.new_data_hits.log_operation")
def test_load_flags_from_redis_success(mock_log, mock_redis):
    redis_mock_instance = MagicMock()
    redis_mock_instance.get.return_value = '{"hits_data": {"x": 1}}'
    mock_redis.return_value = redis_mock_instance

    from scripts.new_data_hits import load_flags_hits
    result = load_flags_hits()

    assert result == {"hits_data": {"x": 1}}
    redis_mock_instance.get.assert_called_once_with('flags:hits_data')
    mock_log.assert_any_call("DEBUG", "Flags loaded from Redis")

@patch("scripts.new_data_hits.get_redis_connection")
@patch("scripts.new_data_hits.log_operation")
@patch("scripts.new_data_hits.save_flags_hits")
@patch("scripts.new_data_hits.os.path.exists")
def test_load_flags_from_redis_empty_no_backup(mock_exists, mock_save, mock_log, mock_redis):

    mock_redis.get.return_value = None  # Redis returns no data
    mock_exists.return_value = False    # No backup file exists

    from scripts.new_data_hits import load_flags_hits
    result = load_flags_hits()

    assert result == {"hits_data": {}}
    mock_save.assert_called_once_with({"hits_data": {}})
    mock_log.assert_any_call("INFO", "Flags not found, initializing empty")

@patch("scripts.new_data_hits.get_redis_connection")
@patch("scripts.new_data_hits.log_operation")
@patch("scripts.new_data_hits.save_flags_hits")
@patch("scripts.new_data_hits.os.path.exists")
@patch("scripts.new_data_hits.json.load")
def test_load_flags_from_backup_file(mock_json_load, mock_exists, mock_save, mock_log, mock_redis):

    mock_redis.get.return_value = None  # Redis returns no data
    mock_exists.return_value = True     # Backup file exists
    mock_json_load.return_value = {"hits_data": {"from_file": 1}}

    from scripts.new_data_hits import load_flags_hits
    result = load_flags_hits()

    assert result == {"hits_data": {"from_file": 1}}
    mock_log.assert_any_call("INFO", "Flags loaded from backup file")