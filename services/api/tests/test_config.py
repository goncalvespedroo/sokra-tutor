from pathlib import Path

import pytest

from app.core.config import Settings


@pytest.fixture(autouse=True)
def clear_openrouter_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in Settings.model_fields:
        if field.startswith("openrouter_"):
            monkeypatch.delenv(f"SOKRA_{field.upper()}", raising=False)


def test_settings_defaults_to_development_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SOKRA_ENVIRONMENT", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.environment == "development"


def test_settings_reads_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SOKRA_ENVIRONMENT", "production")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.environment == "production"


def test_settings_loads_dotenv_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("SOKRA_ENVIRONMENT", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SOKRA_ENVIRONMENT=staging\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.environment == "staging"


def test_settings_openrouter_defaults_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.openrouter_api_key is None
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_chat_model is None
    assert settings.openrouter_embedding_model is None
    assert settings.openrouter_timeout_seconds == 30.0
    assert settings.openrouter_retries == 2


def test_settings_reads_openrouter_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOKRA_OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setenv("SOKRA_OPENROUTER_BASE_URL", "https://example.com/api/v1")
    monkeypatch.setenv("SOKRA_OPENROUTER_CHAT_MODEL", "test/chat-model")
    monkeypatch.setenv("SOKRA_OPENROUTER_EMBEDDING_MODEL", "test/embedding-model")
    monkeypatch.setenv("SOKRA_OPENROUTER_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("SOKRA_OPENROUTER_RETRIES", "4")

    settings = Settings()

    assert settings.openrouter_api_key == "test-only-key"
    assert settings.openrouter_base_url == "https://example.com/api/v1"
    assert settings.openrouter_chat_model == "test/chat-model"
    assert settings.openrouter_embedding_model == "test/embedding-model"
    assert settings.openrouter_timeout_seconds == 45.5
    assert settings.openrouter_retries == 4
