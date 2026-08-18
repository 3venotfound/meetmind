from functools import lru_cache
from pathlib import Path

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
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:5500"

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
