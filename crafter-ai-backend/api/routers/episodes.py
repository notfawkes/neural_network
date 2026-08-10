"""Episodes router."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..schemas.episodes import (
    EpisodeSummary,
    EpisodeDetail,
    EpisodeFramesResponse,
)
from core.config import settings
from services.episode_service import EpisodeService

router = APIRouter(prefix="/api/episodes", tags=["episodes"])
episode_service = EpisodeService(settings.recordings_dir)


@router.get("", response_model=List[EpisodeSummary])
async def list_episodes(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    eps = episode_service.list_episodes(limit=limit, offset=offset)
    return [EpisodeSummary(**ep) for ep in eps]


@router.get("/{episode_id}", response_model=EpisodeDetail)
async def get_episode(episode_id: int):
    ep = episode_service.get_episode_detail(episode_id)
    if not ep:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
    return EpisodeDetail(**ep)


@router.get("/{episode_id}/frames", response_model=EpisodeFramesResponse)
async def get_episode_frames(
    episode_id: int,
    scale: int = Query(4, ge=1, le=8, description="Nearest-neighbor scaling factor"),
):
    frames = episode_service.get_episode_frames(episode_id, scale_factor=scale)
    if frames is None:
        raise HTTPException(status_code=404, detail=f"Frames for episode {episode_id} not found")
    return EpisodeFramesResponse(
        episode_id=episode_id,
        total_frames=len(frames),
        frame_interval=1,
        frames=frames,
    )
