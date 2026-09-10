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


@lru_cache
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache
def get_mongo_settings() -> MongoSettings:
    return MongoSettings()
