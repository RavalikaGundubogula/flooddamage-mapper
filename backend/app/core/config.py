from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration."""

    PROJECT_NAME: str = "Mapping Flood Damage from Space"
    VERSION: str = "0.1.0"
    API_PREFIX: str = ""
    ENV: str = "development"
    DEBUG: bool = True

    # Server config
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Satellite Catalog & Download (Copernicus Data Space Ecosystem)
    COPERNICUS_CATALOG_URL: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    COPERNICUS_DOWNLOAD_URL: str = "https://download.dataspace.copernicus.eu/odata/v1/Products"
    COPERNICUS_AUTH_URL: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    COPERNICUS_USERNAME: Optional[str] = None
    COPERNICUS_PASSWORD: Optional[str] = None
    COPERNICUS_CLIENT_ID: Optional[str] = None
    COPERNICUS_CLIENT_SECRET: Optional[str] = None
    CATALOG_TIMEOUT_SECONDS: int = 20
    CACHE_ENABLED: bool = True

    # OpenStreetMap Overpass API (Module 5 Pre-event Infrastructure)
    OVERPASS_URL: str = "https://overpass-api.de/api/interpreter"
    OVERPASS_TIMEOUT_SECONDS: int = 25
    DATA_OSM_CACHE_DIR: str = "data/cache/osm"

    # Data storage paths
    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"
    DATA_OUTPUTS_DIR: str = "data/outputs"
    DATA_CACHE_DIR: str = "data/cache"
    KUROSIWO_DATA_DIR: str = "data/datasets/kuro_siwo"
    MODEL_ARTIFACTS_DIR: str = "data/models"

    model_config = SettingsConfigDict(
        env_file=(
            Path(__file__).resolve().parent.parent.parent / ".env",
            Path(__file__).resolve().parent.parent.parent.parent / ".env",
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_data_path(self, dir_name: str) -> Path:
        """Helper to get resolved Path for data directories."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        return base_dir / dir_name


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
