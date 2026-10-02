from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.dropout_risk import router as dropout_risk_router
from app.api.health import router as health_router
from app.api.summary import router as summary_router
from app.api.recommend import router as recommend_router
from app.api.knowledge import router as knowledge_router


app = FastAPI(title="Online Study Platform AI")
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(summary_router)
app.include_router(recommend_router)
app.include_router(dropout_risk_router)
app.include_router(knowledge_router)
