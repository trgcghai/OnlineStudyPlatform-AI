from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import Session

from app.db.mongo import get_mongo_database
from app.db.session import get_engine


router = APIRouter()

ComponentStatus = Literal["UP", "DOWN"]


class ComponentHealth(BaseModel):
    status: ComponentStatus


class HealthResponse(BaseModel):
    status: ComponentStatus
    components: dict[str, ComponentHealth]


class RootResponse(BaseModel):
    service: str
    status: ComponentStatus


def check_postgresql() -> ComponentHealth:
    with Session(get_engine()) as session:
        session.exec(text("SELECT 1"))

    return ComponentHealth(status="UP")


async def check_mongodb() -> ComponentHealth:
    await get_mongo_database().command("ping")

    return ComponentHealth(status="UP")


@router.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    return RootResponse(service="online-study-platform-ai", status="UP")


@router.get("/health", response_model=HealthResponse)
async def read_health(response: Response) -> HealthResponse:
    try:
        postgresql = check_postgresql()
    except Exception:
        postgresql = ComponentHealth(status="DOWN")

    try:
        mongodb = await check_mongodb()
    except Exception:
        mongodb = ComponentHealth(status="DOWN")

    components = {"postgresql": postgresql, "mongodb": mongodb}
    service_status: ComponentStatus = "UP"
    if any(component.status == "DOWN" for component in components.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        service_status = "DOWN"

    return HealthResponse(status=service_status, components=components)
