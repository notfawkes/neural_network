"""Pydantic schemas for models and checkpoints."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CheckpointInfo(BaseModel):
    model_id: str
    filename: str
    path: str
    step: int
    tag: str
    episode_count: int
    model_type: str
    size_bytes: int
    created_at: float


class CompareRequest(BaseModel):
    checkpoint_a: str = Field(description="ID or path of checkpoint A")
    checkpoint_b: str = Field(description="ID or path of checkpoint B")
    obs_source_episode_id: Optional[int] = Field(default=None, description="Optional episode ID to sample frames from")
    num_samples: int = Field(default=4, ge=1, le=16)


class ActionComparison(BaseModel):
    action_index: int
    action_name: str
    prob_a: float
    prob_b: float
    delta: float


class SampleComparison(BaseModel):
    sample_index: int
    frame_base64: Optional[str] = None
    value_a: float
    value_b: float
    entropy_a: float
    entropy_b: float
    top_action_a: str
    top_action_b: str
    actions: List[ActionComparison]


class CompareResponse(BaseModel):
    model_a_info: Dict[str, Any]
    model_b_info: Dict[str, Any]
    samples: List[SampleComparison]
