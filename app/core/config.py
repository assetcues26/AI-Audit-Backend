from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    VLM_MODEL: str = "gemini-3.1-flash-lite-preview"
    YOLO_MODEL: str = "yolov8n.pt"
    STORAGE_PROVIDER: str = "local"
    LOCAL_STORAGE_PATH: str = "./audit_sessions"
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
