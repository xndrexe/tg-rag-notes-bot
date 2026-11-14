"""Конфигурация бота из переменных окружения."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения. Источник — переменные окружения и `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: str = Field(..., alias="BOT_TOKEN")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    store_dir: Path = Field(default=Path("./data/vectors"), alias="STORE_DIR")
    storage_dir: Path = Field(default=Path("./data/docs"), alias="STORAGE_DIR")

    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL",
    )

    chunk_size: int = Field(default=600, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=80, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")

    max_file_mb: int = Field(default=20, alias="MAX_FILE_MB")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def ensure_dirs(self) -> None:
        """Создаёт рабочие директории если их нет."""
        for path in (self.data_dir, self.store_dir, self.storage_dir):
            path.mkdir(parents=True, exist_ok=True)
