from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(REPO_ROOT / ".env", ".env"), extra="ignore")

    database_url: str = "postgresql+psycopg://hairstyle:change-me@localhost:5432/hairstyle_advisor"
    image_storage_dir: Path = Path("./storage/images")
    max_upload_bytes: int = 8 * 1024 * 1024
    cors_origins: list[str] = ["http://localhost:3000"]
    use_hair_segmenter: bool = True


settings = Settings()
