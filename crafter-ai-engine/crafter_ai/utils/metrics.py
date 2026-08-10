"""
Metrics tracker and Crafter score computation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
from pathlib import Path

from ..env.adapter import ACHIEVEMENT_NAMES, NUM_ACHIEVEMENTS


def compute_crafter_score(success_rates: Dict[str, float]) -> float:
    """
    Compute the Crafter geometric mean score.

    Score = geometric mean of success rates (clipped to [0, 1]).

    Parameters
    ----------
    success_rates : dict
        {achievement_name: success_rate} where success_rate ∈ [0, 1].

    Returns
    -------
    float : Score in [0, 1].
    """
    rates = []
    for name in ACHIEVEMENT_NAMES:
        rate = success_rates.get(name, 0.0)
        rate = max(0.0, min(1.0, float(rate)))
        rates.append(rate + 1e-6)  # Add small epsilon for log stability

    return float(np.exp(np.mean(np.log(rates))))


class MetricsTracker:
    """
    Tracks training metrics and episode statistics.

    Thread-safe enough for single-process use.

    Parameters
    ----------
    log_dir : str
        Directory to save metrics.
    experiment_name : str
    """

    def __init__(self, log_dir: str = "logs", experiment_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Episode tracking
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_steps: List[int] = []  # global step when episode ended

        # Achievement tracking: for each achievement, list of bools per episode
        self._achievement_records: Dict[str, List[bool]] = defaultdict(list)

        # Training update metrics
        self._update_metrics: List[Dict[str, float]] = []

        # Windowed averages
        self._window = 100

    def record_episode(
        self,
        reward: float,
        length: int,
        global_step: int,
        achievements: Optional[Dict[str, bool]] = None,
    ):
        """Record a completed episode."""
        self._episode_rewards.append(float(reward))
        self._episode_lengths.append(int(length))
        self._episode_steps.append(global_step)

        if achievements:
            for name in ACHIEVEMENT_NAMES:
                self._achievement_records[name].append(
                    bool(achievements.get(name, False))
                )

    def record_update(self, metrics: Dict[str, float]):
        """Record a training update's metrics."""
        self._update_metrics.append(metrics)

    def get_episode_stats(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get episode statistics over a sliding window."""
        w = window or self._window
        if not self._episode_rewards:
            return {}

        recent_rewards = self._episode_rewards[-w:]
        recent_lengths = self._episode_lengths[-w:]

        return {
            "mean_reward": float(np.mean(recent_rewards)),
            "std_reward": float(np.std(recent_rewards)),
            "max_reward": float(np.max(recent_rewards)),
            "min_reward": float(np.min(recent_rewards)),
            "mean_length": float(np.mean(recent_lengths)),
            "total_episodes": len(self._episode_rewards),
        }

    def get_achievement_rates(self) -> Dict[str, float]:
        """Compute achievement success rates across all episodes."""
        rates = {}
        for name in ACHIEVEMENT_NAMES:
            records = self._achievement_records.get(name, [])
            if records:
                rates[name] = float(np.mean(records))
            else:
                rates[name] = 0.0
        return rates

    def get_crafter_score(self) -> float:
        """Compute current Crafter score."""
        return compute_crafter_score(self.get_achievement_rates())

    def get_update_stats(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get recent update metrics."""
        w = window or 10
        if not self._update_metrics:
            return {}
        recent = self._update_metrics[-w:]
        keys = set(k for m in recent for k in m)
        return {
            k: float(np.mean([m[k] for m in recent if k in m]))
            for k in keys
        }

    def get_full_summary(self) -> Dict[str, Any]:
        """Return complete metrics summary."""
        return {
            "episode_stats": self.get_episode_stats(),
            "achievement_rates": self.get_achievement_rates(),
            "crafter_score": self.get_crafter_score(),
            "update_stats": self.get_update_stats(),
            "total_episodes": len(self._episode_rewards),
            "total_steps": self._episode_steps[-1] if self._episode_steps else 0,
        }

    def save(self):
        """Save metrics to JSON."""
        path = self.log_dir / f"{self.experiment_name}_metrics.json"
        data = {
            "episode_rewards": self._episode_rewards,
            "episode_lengths": self._episode_lengths,
            "episode_steps": self._episode_steps,
            "achievement_records": {k: v for k, v in self._achievement_records.items()},
            "update_metrics": self._update_metrics[-1000:],  # last 1000 updates
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, experiment_name: str = "experiment") -> "MetricsTracker":
        """Load metrics from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tracker = cls(experiment_name=experiment_name)
        tracker._episode_rewards = data.get("episode_rewards", [])
        tracker._episode_lengths = data.get("episode_lengths", [])
        tracker._episode_steps = data.get("episode_steps", [])
        tracker._achievement_records = defaultdict(
            list, data.get("achievement_records", {})
        )
        tracker._update_metrics = data.get("update_metrics", [])
        return tracker

