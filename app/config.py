from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None
    s3_prefix: str = "garments/3d-models"
    s3_public_base_url: Optional[str] = None
    s3_use_presigned_url: bool = False
    s3_presigned_expiry_seconds: int = 3600
    local_output_dir: Path = Path("output_models")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.local_output_dir.mkdir(parents=True, exist_ok=True)
