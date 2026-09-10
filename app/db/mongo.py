from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_mongo_settings


@lru_cache
def get_mongo_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(get_mongo_settings().mongodb_uri)


def get_mongo_database() -> AsyncIOMotorDatabase:
    return get_mongo_client().get_default_database()
