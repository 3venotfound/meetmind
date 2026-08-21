from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent


def _backend_relative_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = BACKEND_DIR / expanded
    return expanded.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "MeetMind API"
    app_env: str = "development"
    database_path: Path = Path("storage/meetmind.db")
    storage_root: Path = Path("storage")
    max_upload_size_bytes: int = Field(default=524_288_000, gt=0)
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_file_timeout_seconds: float = Field(default=300, gt=0)
    gemini_file_poll_interval_seconds: float = Field(default=2, gt=0)
    ai_timeout_seconds: float = Field(default=900, gt=0)
    cv_timeout_seconds: float = Field(default=1800, gt=0)
    ai_python_executable: str = ""
    cv_python_executable: str = ""
    cors_origins: str = (
        "https://meetmiind.netlify.app,"
        "http://127.0.0.1:5500,"
        "http://localhost:5500"
    )

    @property
    def resolved_database_path(self) -> Path:
        return _backend_relative_path(self.database_path)

    @property
    def resolved_storage_root(self) -> Path:
        return _backend_relative_path(self.storage_root)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
