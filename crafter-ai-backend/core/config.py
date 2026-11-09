"""Backend configuration."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRAFTER_")

    app_name: str = "CRAfter AI Lab API"
    version: str = "0.1.0"
    debug: bool = True

    # Base paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    workspace_dir: Path = base_dir.parent
    engine_dir: Path = workspace_dir / "crafter-ai-engine"
    
    # Data directories
    recordings_dir: Path = workspace_dir / "recordings"
    checkpoints_dir: Path = workspace_dir / "checkpoints"
    experiments_dir: Path = workspace_dir / "experiments"
    state_dir: Path = workspace_dir / "experiments" / "state"
    logs_dir: Path = workspace_dir / "logs"
    figures_dir: Path = workspace_dir / "figures"

    # Training process
    python_bin: str = "python3"
    train_script: Path = workspace_dir / "crafter-ai-engine" / "scripts" / "train.py"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


settings = Settings()

# Ensure required dirs exist
settings.recordings_dir.mkdir(parents=True, exist_ok=True)
settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
settings.state_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
settings.figures_dir.mkdir(parents=True, exist_ok=True)
