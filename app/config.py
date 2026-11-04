"""Every setting comes from environment variables (12-factor). Nothing environment-specific or
secret is hard-coded, so the same image runs on a laptop, in Compose and in the cloud."""
import os


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    # local  = load the Keras model inside this process (Task 4)
    # remote = call the model API deployed in the cloud (Tasks 7/8)
    INFERENCE_MODE = os.getenv("INFERENCE_MODE", "local").lower()
    MODEL_PATH = os.getenv("MODEL_PATH", "models/champion.keras")
    THRESHOLD = float(os.environ["THRESHOLD"]) if os.getenv("THRESHOLD") else None
    MODEL_API_URL = os.getenv("MODEL_API_URL", "").rstrip("/")
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "30"))
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
    # Flask rejects bigger bodies before our code runs (+64 KB for multipart overhead)
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024 + 64 * 1024
    DEBUG = _bool("FLASK_DEBUG", False)
    HISTORY_PAGE_SIZE = int(os.getenv("HISTORY_PAGE_SIZE", "20"))
