"""Router for listing, viewing, and generating training figures."""
import os
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from core.config import settings

router = APIRouter(prefix="/api/figures", tags=["figures"])

FIGURE_DESCRIPTIONS = {
    "01_learning_curves.png": "Episode Returns and Survival Steps over training",
    "02_ppo_training_losses.png": "Actor Policy Loss, Critic Value Loss, Entropy, and KL Divergence",
    "03_crafter_score_progression.png": "Crafter Official Benchmark Geometric Mean Score (%)",
    "04_achievement_success_rates.png": "All 22 Crafter Achievement Unlock Rates by Domain",
    "05_achievement_heatmap.png": "Temporal Heatmap of Achievement Discovery & Frequency",
    "06_action_policy_distribution.png": "Policy Discrete Action Frequency Distribution across 17 Actions",
    "07_spatial_exploration_heatmap.png": "2D Agent World Trajectory and Terrain Exploration Heatmap",
    "08_neural_representation_inspection.png": "Raw Visual Observation alongside CNN & LSTM Latent States",
    "09_model_checkpoint_comparison.png": "Return, Survival, and Milestone Comparison across Checkpoints",
    "10_training_summary_dashboard.png": "Multi-Panel Master Summary Dashboard of Entire Training Run",
}


@router.get("", response_model=List[Dict[str, Any]])
async def list_figures():
    """List all available publication-grade training figures."""
    figures = []
    if not settings.figures_dir.exists():
        return []

    for p in sorted(settings.figures_dir.glob("*.png")):
        stat = p.stat()
        desc = FIGURE_DESCRIPTIONS.get(p.name, "Training evaluation plot")
        figures.append({
            "filename": p.name,
            "title": p.stem.replace("_", " ").title(),
            "description": desc,
            "size_bytes": stat.st_size,
            "url": f"/api/figures/download/{p.name}",
            "created_at": stat.st_mtime,
        })
    return figures


@router.get("/download/{filename}")
async def get_figure_image(filename: str):
    """Serve figure image directly."""
    file_path = settings.figures_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(str(file_path), media_type="image/png")


@router.post("/generate")
async def generate_figures():
    """Trigger on-demand generation of all figures from latest logs and recordings."""
    try:
        from crafter_ai.evaluation.figures import FigureGenerator
        gen = FigureGenerator(
            output_dir=str(settings.figures_dir),
            log_dir=str(settings.logs_dir),
            recordings_dir=str(settings.recordings_dir),
            checkpoints_dir=str(settings.checkpoints_dir),
        )
        saved = gen.generate_all_figures(experiment_name="crafter_lstm_ppo")
        return {"status": "success", "generated_count": len(saved), "figures": [p.name for p in saved]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate figures: {e}")

