from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    environment: str = "development"

    openrouter_api_key: str | None = Field(default=None, repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str | None = None
    openrouter_embedding_model: str | None = None
    openrouter_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    # Extra attempts after the initial request.
    openrouter_retries: int = Field(default=2, ge=0)

    model_config = SettingsConfigDict(
        env_prefix="SOKRA_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
