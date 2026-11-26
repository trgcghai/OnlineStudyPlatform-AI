from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_model_candidates(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            candidate.strip()
            for candidate in value.split(",")
            if candidate.strip()
        )
    )


class DatabaseSettings(BaseSettings):
    host: str = Field(validation_alias="DB_HOST")
    port: int = Field(validation_alias="DB_PORT")
    username: str = Field(validation_alias="DB_USERNAME")
    password: str = Field(validation_alias="DB_PASSWORD")
    database: str = Field(validation_alias="DB_NAME")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MongoSettings(BaseSettings):
    mongodb_uri: str = Field(validation_alias="MONGODB_URI")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AiSettings(BaseSettings):
    ai_token: str | None = Field(default=None, validation_alias="AI_TOKEN")
    summary_chunk_size: int = Field(
        default=6000, ge=500, le=30000, validation_alias="SUMMARY_CHUNK_SIZE"
    )
    summary_max_output_tokens: int = Field(
        default=2048, ge=128, le=8192, validation_alias="SUMMARY_MAX_OUTPUT_TOKENS"
    )
    ninerouter_url: str = Field(default="", validation_alias="NINEROUTER_URL")
    ninerouter_api_key: str | None = Field(
        default=None, validation_alias="NINEROUTER_API_KEY"
    )
    ninerouter_stt_model: str = Field(
        default="", validation_alias="NINEROUTER_STT_MODEL"
    )
    ninerouter_summary_model: str = Field(
        default="", validation_alias="NINEROUTER_SUMMARY_MODEL"
    )
    ninerouter_summary_timeout_seconds: float = Field(
        default=120,
        gt=0,
        validation_alias="NINEROUTER_SUMMARY_TIMEOUT_SECONDS",
    )
    ninerouter_chat_model: str = Field(
        default="", validation_alias="NINEROUTER_CHAT_MODEL"
    )
    ninerouter_chat_timeout_seconds: float = Field(
        default=120,
        gt=0,
        validation_alias="NINEROUTER_CHAT_TIMEOUT_SECONDS",
    )
    ninerouter_language: str = Field(
        default="vi", validation_alias="NINEROUTER_LANGUAGE"
    )
    ninerouter_timeout_seconds: float = Field(
        default=1500,
        gt=0,
        validation_alias="NINEROUTER_TIMEOUT_SECONDS",
    )
    transcribe_max_bytes: int = Field(
        default=1_073_741_824, ge=1, validation_alias="TRANSCRIBE_MAX_BYTES"
    )
    transcribe_download_timeout_seconds: float = Field(
        default=60, gt=0, validation_alias="TRANSCRIBE_DOWNLOAD_TIMEOUT_SECONDS"
    )
    transcribe_allowed_hosts: str = Field(
        default="minio,localhost,127.0.0.1", validation_alias="TRANSCRIBE_ALLOWED_HOSTS"
    )
    qdrant_url: str = Field(
        default="http://localhost:6333", validation_alias="QDRANT_URL"
    )
    qdrant_collection: str = Field(
        default="lesson_knowledge", validation_alias="QDRANT_COLLECTION"
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_cache_dir: str = Field(
        default="/var/cache/embeddings", validation_alias="EMBEDDING_CACHE_DIR"
    )
    rag_top_k: int = Field(default=5, ge=1, le=20, validation_alias="RAG_TOP_K")
    knowledge_chunk_size: int = Field(
        default=1200, ge=300, le=4000, validation_alias="KNOWLEDGE_CHUNK_SIZE"
    )
    knowledge_chunk_overlap: int = Field(
        default=200, ge=0, le=1000, validation_alias="KNOWLEDGE_CHUNK_OVERLAP"
    )
    knowledge_max_bytes: int = Field(
        default=26_214_400, ge=1, validation_alias="KNOWLEDGE_MAX_BYTES"
    )
    knowledge_max_characters: int = Field(
        default=2_000_000, ge=1000, validation_alias="KNOWLEDGE_MAX_CHARACTERS"
    )
    knowledge_max_archive_bytes: int = Field(
        default=209_715_200, ge=1, validation_alias="KNOWLEDGE_MAX_ARCHIVE_BYTES"
    )
    knowledge_download_timeout_seconds: float = Field(
        default=60, gt=0, validation_alias="KNOWLEDGE_DOWNLOAD_TIMEOUT_SECONDS"
    )
    knowledge_allowed_hosts: str = Field(
        default="minio,localhost,127.0.0.1", validation_alias="KNOWLEDGE_ALLOWED_HOSTS"
    )
    dropout_model_dir: str = Field(
        default="models/dropout-risk",
        validation_alias="DROPOUT_MODEL_DIR",
    )
    dropout_min_training_samples: int = Field(
        default=10,
        validation_alias="DROPOUT_MIN_TRAINING_SAMPLES",
    )
    minio_endpoint: str = Field(
        default="http://localhost:9000",
        validation_alias="MINIO_ENDPOINT",
    )
    minio_access_key: str = Field(
        default="minioadmin",
        validation_alias="MINIO_ACCESS_KEY",
    )
    minio_secret_key: str = Field(
        default="minioadmin",
        validation_alias="MINIO_SECRET_KEY",
    )
    minio_bucket_ai_models: str = Field(
        default="ai-models",
        validation_alias="MINIO_BUCKET_AI_MODELS",
    )
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    dropout_model_cache_dir: str = Field(
        default="/var/cache/ai-models",
        validation_alias="DROPOUT_MODEL_CACHE_DIR",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def ninerouter_chat_models(self) -> tuple[str, ...]:
        return _parse_model_candidates(self.ninerouter_chat_model)

    @property
    def ninerouter_summary_models(self) -> tuple[str, ...]:
        return _parse_model_candidates(self.ninerouter_summary_model)

    @property
    def ninerouter_stt_models(self) -> tuple[str, ...]:
        return _parse_model_candidates(self.ninerouter_stt_model)

    @property
    def transcribe_allowed_host_set(self) -> set[str]:
        return {
            host.strip().casefold()
            for host in self.transcribe_allowed_hosts.split(",")
            if host.strip()
        }

    @property
    def knowledge_allowed_host_set(self) -> set[str]:
        return {
            host.strip().casefold()
            for host in self.knowledge_allowed_hosts.split(",")
            if host.strip()
        }


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache
def get_mongo_settings() -> MongoSettings:
    return MongoSettings()


@lru_cache
def get_ai_settings() -> AiSettings:
    return AiSettings()
