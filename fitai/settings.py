from pathlib import Path
import os
from urllib.parse import unquote, urlparse

from django.core.exceptions import ImproperlyConfigured

# =========================
# BASE
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# SECURITY
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

# Локально DEBUG=True
# На Render задаємо DEBUG=False через Environment Variables
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = os.environ.get(
    "ALLOWED_HOSTS",
    "127.0.0.1,localhost,.onrender.com"
).split(",")

CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS",
    "https://*.onrender.com"
).split(",")


# =========================
# APPLICATIONS
# =========================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # apps
    "fitness",
]


# =========================
# MIDDLEWARE
# =========================
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",

    # WhiteNoise
    "whitenoise.middleware.WhiteNoiseMiddleware",

    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# =========================
# URLS / WSGI
# =========================
ROOT_URLCONF = "fitai.urls"
WSGI_APPLICATION = "fitai.wsgi.application"


# =========================
# TEMPLATES
# =========================
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


# =========================
# DATABASE
# =========================

def _database_config_from_env():
    database_url = os.getenv("DATABASE_URL", "").strip()
    postgres_values = {
        "NAME": os.getenv("POSTGRES_DB", "").strip(),
        "USER": os.getenv("POSTGRES_USER", "").strip(),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
        "HOST": os.getenv("POSTGRES_HOST", "localhost").strip(),
        "PORT": os.getenv("POSTGRES_PORT", "5432").strip(),
    }

    if database_url:
        parsed = urlparse(database_url)
        if parsed.scheme in {"postgres", "postgresql"}:
            if not parsed.hostname or not parsed.path.lstrip("/"):
                raise ImproperlyConfigured("DATABASE_URL is missing host or database")
            return {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": unquote(parsed.path.lstrip("/")),
                "USER": unquote(parsed.username or ""),
                "PASSWORD": unquote(parsed.password or ""),
                "HOST": parsed.hostname,
                "PORT": str(parsed.port or 5432),
                "CONN_MAX_AGE": 60,
            }
        if parsed.scheme == "sqlite":
            sqlite_path = unquote(parsed.path)
            return {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": sqlite_path or BASE_DIR / "db.sqlite3",
            }
        raise ImproperlyConfigured(
            "DATABASE_URL must use postgresql://, postgres://, or sqlite://"
        )

    if postgres_values["NAME"]:
        missing = [
            key
            for key in ("USER", "PASSWORD")
            if not postgres_values[key]
        ]
        if missing:
            raise ImproperlyConfigured(
                "Missing PostgreSQL environment values: "
                + ", ".join(f"POSTGRES_{key}" for key in missing)
            )
        return {
            "ENGINE": "django.db.backends.postgresql",
            **postgres_values,
            "CONN_MAX_AGE": 60,
        }

    if not DEBUG:
        raise ImproperlyConfigured(
            "Production requires DATABASE_URL or POSTGRES_DB, "
            "POSTGRES_USER, and POSTGRES_PASSWORD"
        )

    return {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }


DATABASES = {"default": _database_config_from_env()}


# =========================
# INTERNATIONALIZATION
# =========================
LANGUAGE_CODE = "en-us"

TIME_ZONE = "Europe/Berlin"

USE_I18N = True
USE_TZ = True


# =========================
# STATIC FILES
# =========================
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [
    BASE_DIR / "fitness" / "static",
]

STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"


# =========================
# MEDIA
# =========================
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# =========================
# SECURITY (PRODUCTION)
# =========================
if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True

    X_FRAME_OPTIONS = "DENY"

    # HTTPS redirect only in production
    SECURE_SSL_REDIRECT = True

    # Render reverse proxy support
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")


# =========================
# DEFAULT AUTO FIELD
# =========================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
