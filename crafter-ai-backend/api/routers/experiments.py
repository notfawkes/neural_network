"""Experiments router."""
import yaml
from fastapi import APIRouter
from pathlib import Path
from typing import List, Dict, Any

from core.config import settings

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("")
async def list_experiments():
    experiments = []
    configs_dir = settings.workspace_dir / "crafter-ai-engine" / "configs"
    
    if configs_dir.exists():
        for cfg_file in configs_dir.glob("*.yaml"):
            try:
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f)
                exp = cfg.get("experiment", {})
                experiments.append({
                    "id": cfg_file.stem,
                    "name": exp.get("name", cfg_file.stem),
                    "description": exp.get("description", ""),
                    "model_type": exp.get("model_type", "RecurrentActorCritic"),
                    "config_path": str(cfg_file.relative_to(settings.workspace_dir / "crafter-ai-engine")),
                    "timesteps": cfg.get("training", {}).get("total_timesteps", 1000000),
                })
            except Exception:
                continue

    return experiments
