"""Validated, environment-driven configuration (fail fast at startup)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDERS = {"", "replace-me", "changeme"}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-3.5-flash"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    gemini_api_key: SecretStr | None = None  # secret: never logged or printed

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalize_env(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_level(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _placeholder_means_missing(cls, v):
        if v is None:
            return None
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        return None if raw.strip().lower() in PLACEHOLDERS else v

    @model_validator(mode="after")
    def _guardrails(self):
        if self.app_env == "production" and self.debug:
            raise ValueError("DEBUG must not be enabled in production")
        if self.app_env != "test" and self.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required (set it in .env or the environment)")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once; convert raw validation errors into one clear message."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            name = ".".join(str(p) for p in err["loc"]).upper() or "CONFIG"
            problems.append(f"{name}: {err['msg'].removeprefix('Value error, ')}")
        raise ConfigError("Invalid configuration -> " + "; ".join(problems)) from None
