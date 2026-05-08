"""
Unit Tests — Local Storage Adapter
Tests file-based session persistence.
"""
import json
import os
from unittest.mock import patch

from app.services.storage.local_adapter import LocalStorageAdapter


class TestLocalStorageAdapter:
    def test_creates_directory_on_init(self, tmp_storage_path):
        with patch("app.services.storage.local_adapter.settings") as mock_settings:
            mock_settings.LOCAL_STORAGE_PATH = tmp_storage_path
            LocalStorageAdapter()
            assert os.path.exists(tmp_storage_path)

    def test_save_session_creates_json_file(self, tmp_storage_path):
        with patch("app.services.storage.local_adapter.settings") as mock_settings:
            mock_settings.LOCAL_STORAGE_PATH = tmp_storage_path
            adapter = LocalStorageAdapter()

            test_data = [{"pair_index": 0, "result": "test"}]
            adapter.save_session("session-abc", test_data)

            file_path = os.path.join(tmp_storage_path, "session-abc.json")
            assert os.path.exists(file_path)

    def test_saved_json_content_is_correct(self, tmp_storage_path):
        with patch("app.services.storage.local_adapter.settings") as mock_settings:
            mock_settings.LOCAL_STORAGE_PATH = tmp_storage_path
            adapter = LocalStorageAdapter()

            test_data = [{"pair_index": 0, "name": "Test Asset"}]
            adapter.save_session("session-xyz", test_data)

            file_path = os.path.join(tmp_storage_path, "session-xyz.json")
            with open(file_path, "r") as f:
                loaded = json.load(f)
            assert loaded == test_data

    def test_overwrite_existing_session(self, tmp_storage_path):
        with patch("app.services.storage.local_adapter.settings") as mock_settings:
            mock_settings.LOCAL_STORAGE_PATH = tmp_storage_path
            adapter = LocalStorageAdapter()

            adapter.save_session("session-ow", [{"v": 1}])
            adapter.save_session("session-ow", [{"v": 2}])

            file_path = os.path.join(tmp_storage_path, "session-ow.json")
            with open(file_path, "r") as f:
                loaded = json.load(f)
            assert loaded == [{"v": 2}]

    def test_multiple_sessions(self, tmp_storage_path):
        with patch("app.services.storage.local_adapter.settings") as mock_settings:
            mock_settings.LOCAL_STORAGE_PATH = tmp_storage_path
            adapter = LocalStorageAdapter()

            adapter.save_session("session-1", [{"id": 1}])
            adapter.save_session("session-2", [{"id": 2}])

            files = os.listdir(tmp_storage_path)
            assert "session-1.json" in files
            assert "session-2.json" in files
