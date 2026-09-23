from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    environment: str = "development"

    model_config = SettingsConfigDict(
        env_prefix="SOKRA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
