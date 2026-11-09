"""
PPO Trainer (MLP Baseline)
==========================

Implements Proximal Policy Optimization for the CNN + MLP ActorCritic.

Algorithm:
  1. Collect T steps from B envs (rollout)
  2. Compute GAE advantages
  3. For K epochs, iterate over mini-batches:
     a. Recompute action log_probs and values
     b. Compute probability ratios: ratio = exp(log_π_new - log_π_old)
     c. Compute clipped surrogate: min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)
     d. Policy loss = -mean(clipped_surrogate)
     e. Value loss = 0.5 * mean((V_new - R)²)
     f. Entropy bonus = -coeff * mean(entropy)
     g. Total loss = policy_loss + value_coeff * value_loss + entropy_loss
     h. Clip gradients, Adam step
  4. Reset buffer, repeat

References:
  - Schulman et al. 2017, "Proximal Policy Optimization Algorithms"
  - https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time

from ..models.actor_critic import ActorCritic
from ..env.adapter import VecCrafterEnv
from .rollout_buffer import RolloutBuffer
from ..utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO trainer for the CNN + MLP ActorCritic baseline.

    Parameters
    ----------
    model : ActorCritic
    env : VecCrafterEnv
    config : dict
        Training configuration dictionary.
    device : torch.device
    checkpoint_dir : str
    experiment_name : str
    """

    def __init__(
        self,
        model: ActorCritic,
        env: VecCrafterEnv,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "ppo_mlp",
    ):
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device
        self.experiment_name = experiment_name

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_range = config.get("clip_range", 0.2)
        self.entropy_coeff = config.get("entropy_coefficient", 0.01)
        self.value_coeff = config.get("value_coefficient", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.rollout_steps = config.get("rollout_steps", 128)
        self.num_envs = config.get("num_envs", 4)
        self.total_timesteps = config.get("total_timesteps", 1_000_000)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.target_kl = config.get("target_kl", None)

        # Batch size for mini-batches
        total_rollout = self.rollout_steps * self.num_envs
        self.batch_size = config.get("batch_size", total_rollout // 4)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-5,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            rollout_steps=self.rollout_steps,
            num_envs=self.num_envs,
            obs_shape=env.observation_space_shape,
            num_actions=env.num_actions,
            device=device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, experiment_name)

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.update_count = 0
        self._start_time = time.time()

        # Metrics history (kept lightweight)
        self.metrics_history: List[Dict[str, float]] = []

        # Current obs and dones for rollout
        self._last_obs: Optional[np.ndarray] = None
        self._last_dones: Optional[np.ndarray] = None

    def collect_rollout(self, metrics_callback=None) -> Dict[str, float]:
        """
        Collect rollout_steps * num_envs transitions from the environment.

        Returns episode statistics from completed episodes during rollout.
        """
        self.model.eval()
        self.buffer.reset()

        episode_rewards = []
        episode_lengths = []
        episode_achievements = []

        for step in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(
                self._last_obs, dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                action, log_prob, entropy, value, probs = self.model.get_action_and_value(obs_tensor)

            actions_np = action.cpu().numpy()
            log_probs_np = log_prob.cpu().numpy()
            values_np = value.squeeze(-1).cpu().numpy()

            # Step environments
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            self.buffer.add(
                obs=self._last_obs,
                actions=actions_np,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs_np,
                values=values_np,
            )

            self._last_obs = next_obs
            self._last_dones = dones
            self.global_step += self.num_envs

            # Stream step telemetry and observation
            if metrics_callback:
                raw_obs = self.env.get_raw_obs() if hasattr(self.env, "get_raw_obs") else None
                metrics_callback({
                    "type": "step",
                    "step": self.global_step,
                    "action": int(actions_np[0]),
                    "action_probs": probs[0].cpu().numpy(),
                    "critic_value": float(values_np[0]),
                    "reward": float(rewards[0]),
                    "raw_obs": raw_obs[0] if raw_obs is not None else None,
                    "info": infos[0],
                })

            # Track completed episodes
            for i, info in enumerate(infos):
                if info.get("episode_done", False):
                    ep_reward = info.get("episode_reward", 0.0)
                    ep_length = info.get("episode_step", 0)
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_achievements.append(info.get("achievements", {}))
                    self.episode_count += 1

                    if metrics_callback:
                        metrics_callback({
                            "type": "episode_done",
                            "episode_id": info.get("episode_id", 0),
                            "reward": ep_reward,
                            "length": ep_length,
                            "step": self.global_step,
                        })

        # Compute last values for GAE bootstrap
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(
                self._last_obs, dtype=torch.float32
            ).to(self.device)
            last_values = self.model.get_value(last_obs_tensor).squeeze(-1).cpu().numpy()

        self.buffer.compute_gae(
            last_values=last_values,
            last_dones=self._last_dones,
        )

        stats = {}
        if episode_rewards:
            stats["mean_ep_reward"] = float(np.mean(episode_rewards))
            stats["max_ep_reward"] = float(np.max(episode_rewards))
            stats["min_ep_reward"] = float(np.min(episode_rewards))
            stats["mean_ep_length"] = float(np.mean(episode_lengths))
            stats["num_episodes"] = len(episode_rewards)

        return stats

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on the collected rollout.

        Returns training metrics.
        """
        self.model.train()
        all_metrics = []

        for epoch in range(self.ppo_epochs):
            epoch_metrics = []
            for batch in self.buffer.get_flat_batches(
                batch_size=self.batch_size,
                normalize_advantages=True,
            ):
                metrics = self._update_step(batch)
                epoch_metrics.append(metrics)

                # Early stopping on KL divergence
                if self.target_kl is not None and metrics.get("approx_kl", 0) > self.target_kl:
                    logger.debug(f"Early stopping at KL={metrics['approx_kl']:.4f}")
                    break

            all_metrics.extend(epoch_metrics)
            if not epoch_metrics:
                break

        self.update_count += 1

        # Average metrics
        if all_metrics:
            averaged = {
                k: float(np.mean([m[k] for m in all_metrics if k in m]))
                for k in all_metrics[0]
            }
        else:
            averaged = {}

        averaged["update_count"] = self.update_count
        averaged["global_step"] = self.global_step
        self.metrics_history.append(averaged)

        return averaged

    def _update_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single mini-batch PPO update step."""
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["old_values"]

        # Recompute action log-probs and values
        _, new_log_probs, entropy, new_values, _ = self.model.get_action_and_value(obs, action=actions)

        # Probability ratio: π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Approximate KL (for monitoring only)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

        # Clipped surrogate objective
        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # Value loss with optional clipping
        new_values = new_values.squeeze(-1)
        value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # Entropy bonus
        entropy_loss = -self.entropy_coeff * entropy.mean()

        # Total loss
        loss = policy_loss + self.value_coeff * value_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "grad_norm": float(grad_norm),
        }

    def train(
        self,
        total_timesteps: Optional[int] = None,
        metrics_callback=None,
        checkpoint_interval: int = 50_000,
    ):
        """
        Main training loop.

        Parameters
        ----------
        total_timesteps : int
            Override total timesteps from config.
        metrics_callback : callable
            Called with metric dicts during training.
        checkpoint_interval : int
            Save checkpoint every N steps.
        """
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        # Initialize environments
        logger.info("Resetting environments...")
        self._last_obs, _ = self.env.reset()
        self._last_obs = np.array(self._last_obs)
        self._last_dones = np.zeros(self.num_envs, dtype=np.float32)

        self._start_time = time.time()
        last_checkpoint_step = self.global_step

        logger.info(
            f"Starting PPO training: total_timesteps={total_timesteps}, "
            f"num_envs={self.num_envs}, rollout_steps={self.rollout_steps}"
        )

        while self.global_step < total_timesteps:
            t0 = time.time()

            # Collect rollout
            rollout_stats = self.collect_rollout(metrics_callback=metrics_callback)

            # PPO update
            update_metrics = self.update()
            update_metrics.update(rollout_stats)

            # Speed stats
            elapsed = time.time() - self._start_time
            fps = self.global_step / (elapsed + 1e-8)
            update_metrics["fps"] = fps
            update_metrics["elapsed_seconds"] = elapsed

            # Log
            if self.update_count % 10 == 0:
                logger.info(
                    f"Step {self.global_step}/{total_timesteps} | "
                    f"Episodes: {self.episode_count} | "
                    f"Mean reward: {rollout_stats.get('mean_ep_reward', 0):.2f} | "
                    f"Policy loss: {update_metrics.get('policy_loss', 0):.4f} | "
                    f"Value loss: {update_metrics.get('value_loss', 0):.4f} | "
                    f"FPS: {fps:.0f}"
                )

            if metrics_callback:
                metrics_callback({
                    "type": "update",
                    **update_metrics,
                })

            # Checkpoint
            if self.global_step - last_checkpoint_step >= checkpoint_interval:
                self.save_checkpoint(tag="latest")
                last_checkpoint_step = self.global_step

        # Final checkpoint
        self.save_checkpoint(tag="final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def save_checkpoint(self, tag: str = "latest") -> str:
        """Save model checkpoint and return path."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "update_count": self.update_count,
            "config": self.config,
            "experiment_name": self.experiment_name,
            "model_type": "ActorCritic",
        }
        path = self.checkpoint_manager.save(checkpoint, tag=tag, step=self.global_step)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.update_count = checkpoint.get("update_count", 0)
        logger.info(f"Checkpoint loaded: {path} (step={self.global_step})")

