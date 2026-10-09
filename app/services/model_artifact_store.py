from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from minio import Minio

from app.ml.dropout_constants import MODEL_ARTIFACT_NAME


@dataclass(frozen=True)
class ModelArtifactLocation:
    storage: str
    bucket: str
    object_key: str
    uri: str
    etag: str | None = None


def create_minio_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
) -> Minio:
    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path
    scheme_secure = parsed.scheme == "https"
    return Minio(
        host,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure or scheme_secure,
    )


def build_model_object_key(model_version: str) -> str:
    return f"dropout-risk/{model_version}/{MODEL_ARTIFACT_NAME}"


def cached_model_path(cache_root: Path, model_version: str) -> Path:
    return cache_root / "dropout-risk" / model_version / MODEL_ARTIFACT_NAME


def upload_model_artifact(
    *,
    client: object,
    bucket_name: str,
    model_version: str,
    artifact_path: Path,
) -> ModelArtifactLocation:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    object_key = build_model_object_key(model_version)
    result = client.fput_object(bucket_name, object_key, artifact_path.as_posix())
    uri = f"minio://{bucket_name}/{object_key}"
    return ModelArtifactLocation(
        storage="minio",
        bucket=bucket_name,
        object_key=object_key,
        uri=uri,
        etag=getattr(result, "etag", None),
    )


def download_model_artifact(
    *,
    client: object,
    location: ModelArtifactLocation,
    cache_root: Path,
    model_version: str,
) -> Path:
    destination = cached_model_path(cache_root, model_version)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(
        location.bucket,
        location.object_key,
        destination.as_posix(),
    )
    return destination
