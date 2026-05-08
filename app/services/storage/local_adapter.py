import json
import os

from app.core.config import settings
from app.services.storage.base import StorageProvider


class LocalStorageAdapter(StorageProvider):
    def __init__(self):
        self.path = settings.LOCAL_STORAGE_PATH
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save_session(self, session_id: str, data: dict):
        file_path = os.path.join(self.path, f"{session_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
