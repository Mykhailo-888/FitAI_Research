from fitai import settings


DATABASE_ENV_KEYS = (
    "DATABASE_URL",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
)


def _clear_database_env(monkeypatch):
    for key in DATABASE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_sqlite_fallback_for_debug(monkeypatch):
    _clear_database_env(monkeypatch)
    monkeypatch.setattr(settings, "DEBUG", True)

    config = settings._database_config_from_env()

    assert config["ENGINE"] == "django.db.backends.sqlite3"


def test_database_url_configures_postgresql(monkeypatch):
    _clear_database_env(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://fitai_user:secret@postgres.example:5433/fitai_db",
    )

    config = settings._database_config_from_env()

    assert config == {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "fitai_db",
        "USER": "fitai_user",
        "PASSWORD": "secret",
        "HOST": "postgres.example",
        "PORT": "5433",
        "CONN_MAX_AGE": 60,
    }


def test_individual_postgresql_environment_variables(monkeypatch):
    _clear_database_env(monkeypatch)
    monkeypatch.setenv("POSTGRES_DB", "fitai_db")
    monkeypatch.setenv("POSTGRES_USER", "fitai_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "5432")

    config = settings._database_config_from_env()

    assert config["ENGINE"] == "django.db.backends.postgresql"
    assert config["NAME"] == "fitai_db"
    assert config["HOST"] == "db"
