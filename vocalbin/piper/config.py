from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PIPER_", env_file=".env", extra="ignore"
    )

    model_path: Path
    config_path: Path | None = None
