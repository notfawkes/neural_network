"""
Checkpoint Manager
==================
Saves and loads model checkpoints with version tracking.
"""

import os
import json
import torch
import glob
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints.

    Saves checkpoints with step numbers and maintains a JSON index.
    Supports:
      - Saving with named tags (latest, best, final)
      - Saving with step numbers
      - Loading the best/latest checkpoint
      - Listing all checkpoints
      - Automatic pruning of old checkpoints

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints.
    experiment_name : str
        Prefix for checkpoint files.
    max_checkpoints : int
        Maximum number of numbered checkpoints to keep.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        max_checkpoints: int = 10,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints
        self._index_path = self.checkpoint_dir / f"{experiment_name}_index.json"
        self._index: Dict[str, Any] = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {"checkpoints": [], "best_step": 0, "best_reward": float("-inf")}

    def _save_index(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def save(
        self,
        checkpoint: Dict[str, Any],
        tag: str = "latest",
        step: int = 0,
    ) -> str:
        """
        Save a checkpoint.

        Parameters
        ----------
        checkpoint : dict
            Checkpoint data.
        tag : str
            Tag for this checkpoint: 'latest', 'best', 'final', or 'step_N'.
        step : int
            Training step number.

        Returns
        -------
        str : Path to saved checkpoint.
        """
        # Save tagged checkpoint (overwrites)
        tag_path = self.checkpoint_dir / f"{self.experiment_name}_{tag}.pt"
        torch.save(checkpoint, tag_path)

        # Also save a numbered checkpoint for history
        numbered_path = self.checkpoint_dir / f"{self.experiment_name}_step{step:08d}.pt"
        torch.save(checkpoint, numbered_path)

        # Update index
        entry = {
            "path": str(numbered_path),
            "step": step,
            "tag": tag,
            "episode_count": checkpoint.get("episode_count", 0),
        }
        self._index["checkpoints"].append(entry)

        # Prune old numbered checkpoints
        self._prune_old_checkpoints()
        self._save_index()

        return str(tag_path)

    def _prune_old_checkpoints(self):
        """Remove old numbered checkpoints beyond max_checkpoints."""
        numbered = [
            c for c in self._index["checkpoints"]
            if "step" in c["path"] and "step0" not in c.get("tag", "")
        ]
        # Keep the most recent max_checkpoints
        if len(numbered) > self.max_checkpoints:
            to_remove = numbered[: len(numbered) - self.max_checkpoints]
            for entry in to_remove:
                path = Path(entry["path"])
                if path.exists():
                    path.unlink()
                self._index["checkpoints"].remove(entry)

    def load_latest(self, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
        """Load the latest checkpoint into a model."""
        path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"No latest checkpoint found at {path}")
        return self.load(str(path), model, device)

    def load(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint from path."""
        if device is None:
            device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=device)
        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint: {path}")
        return checkpoint

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return list of all checkpoints."""
        return self._index["checkpoints"]

    def get_latest_path(self) -> Optional[str]:
        """Return path to latest checkpoint."""
        path = self.checkpoint_dir / f"{self.experiment_name}_latest.pt"
        return str(path) if path.exists() else None

    def get_best_path(self) -> Optional[str]:
        """Return path to best checkpoint."""
        path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        return str(path) if path.exists() else None

    def update_best(self, checkpoint: Dict[str, Any], reward: float, step: int):
        """Save as best if reward is better."""
        if reward > self._index.get("best_reward", float("-inf")):
            self._index["best_reward"] = reward
            self._index["best_step"] = step
            self.save(checkpoint, tag="best", step=step)
            logger.info(f"New best checkpoint! Reward={reward:.2f} at step {step}")

