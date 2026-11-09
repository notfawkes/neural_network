"""Pydantic schemas for episodes."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class EpisodeSummary(BaseModel):
    episode_id: int
    global_step: int
    total_reward: float
    length: int
    terminated: bool
    truncated: bool
    num_achievements: int
    achievements: Dict[str, bool] = Field(default_factory=dict)
    duration_seconds: float
    experiment_name: str
    model_type: str
    has_recording: bool = True


class EpisodeDetail(EpisodeSummary):
    actions: List[int] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)
    player_positions: List[List[int]] = Field(default_factory=list)
    critic_values: List[Optional[float]] = Field(default_factory=list)
    action_probs: List[Optional[List[float]]] = Field(default_factory=list)
    lstm_h_norms: List[Optional[float]] = Field(default_factory=list)


class EpisodeFramesResponse(BaseModel):
    episode_id: int
    total_frames: int
    frame_interval: int
    frames: List[str] = Field(description="Base64 encoded JPEG frames")

