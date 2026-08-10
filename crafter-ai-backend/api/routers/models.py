"""Models router."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ..schemas.models import (
    CheckpointInfo,
    CompareRequest,
    CompareResponse,
)
from core.config import settings
from services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])
model_service = ModelService(settings.checkpoints_dir, settings.recordings_dir)


@router.get("", response_model=List[CheckpointInfo])
async def list_models():
    models = model_service.list_checkpoints()
    return [CheckpointInfo(**m) for m in models]


@router.get("/{model_id}")
async def get_model(model_id: str):
    detail = model_service.get_checkpoint_detail(model_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return detail


@router.post("/compare", response_model=CompareResponse)
async def compare_models(req: CompareRequest):
    try:
        res = model_service.compare_checkpoints(
            checkpoint_a=req.checkpoint_a,
            checkpoint_b=req.checkpoint_b,
            obs_source_episode_id=req.obs_source_episode_id,
            num_samples=req.num_samples,
        )
        return CompareResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
