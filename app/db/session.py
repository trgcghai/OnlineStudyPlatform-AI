from functools import lru_cache
from urllib.parse import quote_plus

from sqlalchemy.engine import Engine
from sqlmodel import create_engine

from app.core.config import DatabaseSettings, get_database_settings


def build_database_url(settings: DatabaseSettings) -> str:
    username = quote_plus(settings.username)
    password = quote_plus(settings.password)
    database = quote_plus(settings.database)

    return (
        f"postgresql+psycopg://{username}:{password}"
        f"@{settings.host}:{settings.port}/{database}"
    )


@lru_cache
def get_engine() -> Engine:
    return create_engine(build_database_url(get_database_settings()), pool_pre_ping=True)
