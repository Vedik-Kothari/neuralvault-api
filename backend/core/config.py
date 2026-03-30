# =====================================================
# core/config.py
# Central configuration — reads from .env file.
#
# WHY use a Settings class instead of os.getenv()?
# - Type validation: if SUPABASE_URL is missing, you get
#   a clear error at startup, not a cryptic failure later
# - Single source of truth: import settings anywhere
# - Pydantic validates types automatically
# =====================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from loguru import logger


class Settings(BaseSettings):
    """
    All application settings loaded from environment variables.
    Pydantic automatically reads from .env file.
    """

    # --- Supabase ---
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    supabase_jwt_secret: str
    @property
    def supabase_auth_url(self) -> str:
          """Derived automatically from supabase_url — no .env entry needed."""
          return f"{self.supabase_url}/auth/v1"

    # --- App ---
    app_env: str = "development"
    app_secret_key: str = "change-me"
    # Comma-separated list of allowed CORS origins
    allowed_origins: str = "http://localhost:8501,http://localhost:3000,https://*.vercel.app"

    # --- LLM (used in Phase 5) ---
    groq_api_key: str = ""

    # --- RAG settings ---
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_retrieval_chunks: int = 5
    similarity_threshold: float = 0.3

    # Tell Pydantic to read from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,    # SUPABASE_URL == supabase_url
        extra="ignore",          # ignore unknown env vars
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        """Convert comma-separated string to list."""
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


# lru_cache means Settings is only created ONCE (singleton)
# Calling get_settings() 100 times = same object, no re-reading .env
@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    logger.info(f"Settings loaded | env={settings.app_env}")
    return settings


# Convenience: import `settings` directly anywhere
settings = get_settings()