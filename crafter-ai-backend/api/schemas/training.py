"""Pydantic schemas for training."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class TrainingStartRequest(BaseModel):
    config_path: str = Field(default="configs/default.yaml", description="Path to YAML config")
    resume_checkpoint: Optional[str] = Field(default=None, description="Path to checkpoint to resume from")
    total_timesteps: Optional[int] = Field(default=None, description="Override total timesteps")
    experiment_name: Optional[str] = Field(default=None, description="Override experiment name")


class TrainingResumeRequest(BaseModel):
    checkpoint_path: str = Field(description="Path to checkpoint to resume from")
    total_timesteps: Optional[int] = Field(default=None, description="Override total timesteps")


class TrainingStatusResponse(BaseModel):
    status: str = Field(default="idle", description="Current status: idle, starting, training, stopped, error")
    is_running: bool = Field(default=False)
    pid: Optional[int] = None
    global_step: int = 0
    episode_count: int = 0
    update_count: int = 0
    mean_reward: float = 0.0
    mean_ep_length: float = 0.0
    crafter_score: float = 0.0
    fps: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    elapsed_seconds: float = 0.0
    experiment_name: str = "default"
    latest_checkpoint: Optional[str] = None
    last_update: Optional[float] = None


class MetricsResponse(BaseModel):
    experiment_name: str
    total_steps: int = 0
    total_episodes: int = 0
    crafter_score: float = 0.0
    episode_stats: Dict[str, float] = Field(default_factory=dict)
    achievement_rates: Dict[str, float] = Field(default_factory=dict)
    update_stats: Dict[str, float] = Field(default_factory=dict)
    recent_rewards: List[float] = Field(default_factory=list)
    recent_lengths: List[int] = Field(default_factory=list)
    recent_losses: List[Dict[str, Any]] = Field(default_factory=list)

