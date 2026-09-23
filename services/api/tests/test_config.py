from pathlib import Path

import pytest

from app.core.config import Settings


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
