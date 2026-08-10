"""
Evaluator
=========
Evaluates trained agents on the Crafter environment.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..env.adapter import CrafterEnv, ACHIEVEMENT_NAMES
from ..models.actor_critic import ActorCritic, RecurrentActorCritic
from ..utils.metrics import MetricsTracker, compute_crafter_score

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a trained model over multiple episodes.

    Parameters
    ----------
    model : ActorCritic or RecurrentActorCritic
    device : torch.device
    num_episodes : int
        Number of evaluation episodes.
    seed : int
        Evaluation seed.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        device: torch.device,
        num_episodes: int = 20,
        seed: int = 1000,
    ):
        self.model = model
        self.device = device
        self.num_episodes = num_episodes
        self.seed = seed
        self._is_recurrent = isinstance(model, RecurrentActorCritic)

    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation episodes and compute all metrics.

        Returns
        -------
        dict with:
          - episode_rewards: list of floats
          - episode_lengths: list of ints
          - achievement_rates: dict of achievement_name → success_rate
          - crafter_score: float
          - mean_reward: float
          - std_reward: float
        """
        self.model.eval()
        env = CrafterEnv(reward=True, seed=self.seed, record_raw=False)

        all_rewards = []
        all_lengths = []
        all_achievements = {name: [] for name in ACHIEVEMENT_NAMES}

        with torch.no_grad():
            for ep in range(self.num_episodes):
                episode_reward, episode_length, achievements = self._run_episode(env)
                all_rewards.append(episode_reward)
                all_lengths.append(episode_length)
                for name in ACHIEVEMENT_NAMES:
                    all_achievements[name].append(bool(achievements.get(name, False)))

                if verbose:
                    logger.info(
                        f"Eval episode {ep+1}/{self.num_episodes}: "
                        f"reward={episode_reward:.2f}, length={episode_length}"
                    )

        # Compute achievement success rates
        achievement_rates = {
            name: float(np.mean(all_achievements[name]))
            for name in ACHIEVEMENT_NAMES
        }
        crafter_score = compute_crafter_score(achievement_rates)

        results = {
            "episode_rewards": all_rewards,
            "episode_lengths": all_lengths,
            "achievement_rates": achievement_rates,
            "crafter_score": crafter_score,
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_length": float(np.mean(all_lengths)),
            "num_episodes": self.num_episodes,
        }

        if verbose:
            logger.info(f"Evaluation complete:")
            logger.info(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            logger.info(f"  Mean length: {results['mean_length']:.0f}")
            logger.info(f"  Crafter score: {crafter_score*100:.1f}%")
            for name, rate in sorted(achievement_rates.items(), key=lambda x: -x[1]):
                if rate > 0:
                    logger.info(f"  {name}: {rate*100:.1f}%")

        return results

    def _run_episode(self, env: CrafterEnv):
        """Run a single evaluation episode."""
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        if self._is_recurrent:
            lstm_state = self.model.initial_state(1, self.device)
            done_tensor = torch.zeros(1, device=self.device)

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            if self._is_recurrent:
                action, _, _, _, _, lstm_state = self.model.get_action_and_value(
                    obs_tensor, done_tensor, lstm_state
                )
                # Use greedy action for eval
                logits, _, _, _ = self.model.forward_sequence(
                    obs_tensor.unsqueeze(0),
                    done_tensor.unsqueeze(0),
                    lstm_state,
                )
                action = logits.squeeze(0).squeeze(0).argmax(dim=-1, keepdim=True)
            else:
                logits, _, _ = self.model(obs_tensor)
                action = logits.argmax(dim=-1)

            action_int = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if self._is_recurrent:
                done_tensor = torch.tensor([float(done)], device=self.device)

        achievements = info.get("achievements", {})
        return episode_reward, episode_length, achievements

