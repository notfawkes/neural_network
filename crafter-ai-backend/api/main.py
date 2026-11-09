"""
FastAPI Application Entry Point
================================
CRAfter AI — Visual Reinforcement Learning Lab Backend Server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    training_router,
    episodes_router,
    models_router,
    achievements_router,
    experiments_router,
    websocket_router,
    figures_router,
)
from core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Interactive Reinforcement Learning Visual Lab for Crafter",
)

# CORS Middleware (supports local development on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST and WebSocket routers
app.include_router(training_router)
app.include_router(episodes_router)
app.include_router(models_router)
app.include_router(achievements_router)
app.include_router(experiments_router)
app.include_router(websocket_router)
app.include_router(figures_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.host, port=settings.port, reload=True)
