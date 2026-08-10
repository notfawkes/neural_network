"""Training endpoints router."""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from ..schemas.training import (
    TrainingStartRequest,
    TrainingResumeRequest,
    TrainingStatusResponse,
    MetricsResponse,
)
from core.config import settings
from core.process_manager import ProcessManager
from services.metrics_service import MetricsService
from services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])

# Singletons initialized in main or via dependencies
pm = ProcessManager(settings.workspace_dir, settings.logs_dir)
ms = MetricsService(settings.workspace_dir)
training_service = TrainingService(pm, ms)


@router.post("/start", response_model=Dict[str, Any])
async def start_training(req: TrainingStartRequest):
    try:
        res = training_service.start(
            config_path=req.config_path,
            resume_checkpoint=req.resume_checkpoint,
            total_timesteps=req.total_timesteps,
            experiment_name=req.experiment_name,
        )
        return res
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=Dict[str, Any])
async def stop_training():
    try:
        return training_service.stop()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=Dict[str, Any])
async def resume_training(req: TrainingResumeRequest):
    try:
        return training_service.start(
            resume_checkpoint=req.checkpoint_path,
            total_timesteps=req.total_timesteps,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TrainingStatusResponse)
async def get_status():
    status_data = training_service.get_status()
    return TrainingStatusResponse(**status_data)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    metrics_data = training_service.get_metrics()
    return MetricsResponse(**metrics_data)


@router.get("/logs")
async def get_logs(lines: int = 50):
    return {"logs": pm.get_recent_logs(lines)}
