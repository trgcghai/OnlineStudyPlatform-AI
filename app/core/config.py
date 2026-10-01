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
    gemini_model: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    gemini_timeout_seconds: float = Field(default=8, validation_alias="GEMINI_TIMEOUT_SECONDS")
    dropout_model_dir: str = Field(
        default="models/dropout-risk",
        validation_alias="DROPOUT_MODEL_DIR",
    )
    dropout_min_training_samples: int = Field(
        default=10,
        validation_alias="DROPOUT_MIN_TRAINING_SAMPLES",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache
def get_mongo_settings() -> MongoSettings:
    return MongoSettings()


@lru_cache
def get_ai_settings() -> AiSettings:
    return AiSettings()
