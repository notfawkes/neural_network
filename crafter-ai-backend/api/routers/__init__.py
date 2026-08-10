"""API Routers package."""
from .training import router as training_router
from .episodes import router as episodes_router
from .models import router as models_router
from .achievements import router as achievements_router
from .experiments import router as experiments_router
from .websocket import router as websocket_router

__all__ = [
    "training_router",
    "episodes_router",
    "models_router",
    "achievements_router",
    "experiments_router",
    "websocket_router",
]
