"""Environment-backed settings; secrets never enter frontend responses or logs."""

from __future__ import annotations

from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AIC_", extra="ignore")

    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    artifact_root: Path = Path("/artifacts")
    image_root: Path = Path("/artifacts")
    device: str = "cpu"
    cors_origins: list[AnyHttpUrl | str] = Field(default_factory=lambda: ["http://localhost:5173"])
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-2024-11-20"
    openai_timeout_s: float = Field(default=30.0, gt=0)
    enable_image_answers: bool = True
    collection_suffix: str = "current"
    ocr_jobs_enabled: bool = False
    ocr_data_root: Path = Path("data/prepared")
    ocr_cache_root: Path = Path("artifacts/models")
    ocr_config_path: Path = Path("offline/configs/offline/ocr_ppocrv6.yaml")
