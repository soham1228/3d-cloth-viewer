"""AWS S3 upload helper for generated .glb 3D garment models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config

from app.config import settings


@dataclass
class S3UploadResult:
    bucket: str
    key: str
    s3_uri: str
    model_url: str


def is_s3_configured() -> bool:
    return bool(settings.s3_bucket_name)


def _create_s3_client():
    kwargs = {"region_name": settings.aws_region, "config": Config(signature_version="s3v4")}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def upload_glb_to_s3(
    glb_bytes: bytes,
    filename: str,
    prefix: Optional[str] = None,
) -> S3UploadResult:
    """Upload raw .glb bytes to S3 with model/gltf-binary content type."""
    if not settings.s3_bucket_name:
        raise ValueError("S3_BUCKET_NAME is not configured. Set it in .env or environment variables.")

    clean_prefix = (prefix if prefix is not None else settings.s3_prefix).strip("/")
    key = f"{clean_prefix}/{filename}" if clean_prefix else filename

    client = _create_s3_client()
    client.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=glb_bytes,
        ContentType="model/gltf-binary",
        CacheControl="public, max-age=31536000",
    )

    if settings.s3_use_presigned_url:
        model_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_name, "Key": key},
            ExpiresIn=settings.s3_presigned_expiry_seconds,
        )
    elif settings.s3_public_base_url:
        base = settings.s3_public_base_url.rstrip("/")
        model_url = f"{base}/{key}"
    else:
        model_url = f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{key}"

    return S3UploadResult(
        bucket=settings.s3_bucket_name,
        key=key,
        s3_uri=f"s3://{settings.s3_bucket_name}/{key}",
        model_url=model_url,
    )
