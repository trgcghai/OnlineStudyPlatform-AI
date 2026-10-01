from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_provider: str = Field(default="gemini", validation_alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(
        default="gemini-2.5-flash", validation_alias="GEMINI_MODEL"
    )
    gemini_timeout_seconds: float = Field(
        default=30, validation_alias="GEMINI_TIMEOUT_SECONDS"
    )
    summary_provider: str = Field(default="auto", validation_alias="SUMMARY_PROVIDER")
    summary_chunk_size: int = Field(
        default=6000, ge=500, le=30000, validation_alias="SUMMARY_CHUNK_SIZE"
    )
    summary_max_output_tokens: int = Field(
        default=2048, ge=128, le=8192, validation_alias="SUMMARY_MAX_OUTPUT_TOKENS"
    )
    whisper_model: str = Field(default="small", validation_alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(
        default="int8", validation_alias="WHISPER_COMPUTE_TYPE"
    )
    whisper_language: str = Field(default="vi", validation_alias="WHISPER_LANGUAGE")
    whisper_download_root: str | None = Field(
        default=None, validation_alias="WHISPER_DOWNLOAD_ROOT"
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
