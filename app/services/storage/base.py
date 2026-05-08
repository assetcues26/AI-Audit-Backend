from abc import ABC, abstractmethod

class StorageProvider(ABC):
    @abstractmethod
    def save_session(self, session_id: str, data: dict):
        pass
