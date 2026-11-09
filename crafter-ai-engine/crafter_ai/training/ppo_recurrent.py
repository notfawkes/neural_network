"""
Recurrent PPO Trainer (CNN + LSTM)
====================================

Extends PPO to handle LSTM hidden states correctly.

Key differences from MLP PPO:
  1. LSTM state (h, c) is maintained across rollout steps
  2. At episode boundaries (done=True), LSTM state is zeroed
  3. Mini-batches preserve TEMPORAL ORDER within sequences
  4. Full environment sequences (T, B, ...) are used for LSTM forward passes
  5. Initial LSTM state at the start of each sequence is recorded and replayed
     during the update phase

This implements the approach used in IMPALA and similar recurrent RL papers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
import time

from ..models.actor_critic import RecurrentActorCritic, LSTMState
from ..env.adapter import VecCrafterEnv
from .rollout_buffer import RolloutBuffer
from ..utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class RecurrentPPOTrainer:
    """
    PPO trainer for the CNN + LSTM RecurrentActorCritic.

    Parameters
    ----------
    model : RecurrentActorCritic
    env : VecCrafterEnv
    config : dict
    device : torch.device
    checkpoint_dir : str
    experiment_name : str
    """

    def __init__(
        self,
        model: RecurrentActorCritic,
        env: VecCrafterEnv,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "ppo_lstm",
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
        self.lstm_hidden = config.get("lstm_hidden_size", 256)

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

        # Store initial LSTM state at the START of each rollout
        # (needed to replay LSTM during update)
        self._rollout_initial_lstm_state: Optional[LSTMState] = None

        # Current LSTM state (maintained across steps)
        self._lstm_state: Optional[LSTMState] = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, experiment_name)

        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.update_count = 0
        self._start_time = time.time()

        # Metrics history
        self.metrics_history: List[Dict[str, float]] = []

        # Current obs
        self._last_obs: Optional[np.ndarray] = None
        self._last_dones: Optional[np.ndarray] = None

    def collect_rollout(self, metrics_callback=None) -> Dict[str, float]:
        """
        Collect rollout_steps * num_envs transitions, maintaining LSTM state.

        LSTM state is preserved between steps within the rollout.
        When a done=True, that environment's LSTM state will be zeroed in
        the NEXT timestep's forward pass (handled by forward_sequence).

        Returns episode statistics.
        """
        self.model.eval()
        self.buffer.reset()

        # Save initial LSTM state for replay during update
        self._rollout_initial_lstm_state = (
            self._lstm_state[0].detach().clone(),
            self._lstm_state[1].detach().clone(),
        )

        episode_rewards = []
        episode_lengths = []

        for step in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(
                self._last_obs, dtype=torch.float32
            ).to(self.device)
            dones_tensor = torch.as_tensor(
                self._last_dones, dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                (
                    action,
                    log_prob,
                    entropy,
                    value,
                    probs,
                    self._lstm_state,
                ) = self.model.get_action_and_value(
                    obs_tensor, dones_tensor, self._lstm_state
                )

            actions_np = action.cpu().numpy()
            log_probs_np = log_prob.cpu().numpy()
            values_np = value.squeeze(-1).cpu().numpy()

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
                with torch.no_grad():
                    cnn_feat = self.model.get_features(obs_tensor[0:1])
                    cnn_norm = float(cnn_feat.norm().item())
                    cnn_sample_vec = cnn_feat[0, :16].cpu().numpy().tolist()

                lstm_h_norm = float(self._lstm_state[0].norm().item()) if self._lstm_state is not None else 1.0
                lstm_c_norm = float(self._lstm_state[1].norm().item()) if self._lstm_state is not None else 1.0
                lstm_sample_vec = self._lstm_state[0][0, 0, :16].cpu().numpy().tolist() if self._lstm_state is not None else [0.0] * 16

                metrics_callback({
                    "type": "step",
                    "step": self.global_step,
                    "rollout_step": step + 1,
                    "rollout_total": self.rollout_steps,
                    "action": int(actions_np[0]),
                    "action_probs": probs[0].cpu().numpy(),
                    "critic_value": float(values_np[0]),
                    "reward": float(rewards[0]),
                    "raw_obs": raw_obs[0] if raw_obs is not None else None,
                    "info": infos[0],
                    "lstm_h_norm": lstm_h_norm,
                    "lstm_c_norm": lstm_c_norm,
                    "lstm_sample_vec": lstm_sample_vec,
                    "cnn_norm": cnn_norm,
                    "cnn_sample_vec": cnn_sample_vec,
                    "phase": "COLLECTING_ROLLOUT",
                    "phase_step": f"{step + 1}/{self.rollout_steps}",
                    "phase_details": f"Collecting transition {(step + 1) * self.num_envs}/{self.rollout_steps * self.num_envs} into rollout buffer",
                })

            # Track completed episodes
            for i, info in enumerate(infos):
                if info.get("episode_done", False):
                    ep_reward = info.get("episode_reward", 0.0)
                    ep_length = info.get("episode_step", 0)
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    self.episode_count += 1

                    if metrics_callback:
                        metrics_callback({
                            "type": "episode_done",
                            "env_id": i,
                            "episode_id": info.get("episode_id", 0),
                            "reward": ep_reward,
                            "length": ep_length,
                            "step": self.global_step,
                            "achievements": info.get("achievements", {}),
                            "lstm_h_norm": float(self._lstm_state[0].norm().item()),
                        })

        # Bootstrap value for GAE
        if metrics_callback:
            metrics_callback({
                "type": "phase_change",
                "phase": "CALCULATING_ADVANTAGES",
                "phase_step": "GAE-λ",
                "phase_details": f"Computing Generalized Advantage Estimation (GAE-λ) with γ={self.gamma}, λ={self.gae_lambda}",
            })

        with torch.no_grad():
            last_obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32).to(self.device)
            last_done_t = torch.as_tensor(self._last_dones, dtype=torch.float32).to(self.device)
            last_values, _ = self.model.get_value(last_obs_t, last_done_t, self._lstm_state)
            last_values_np = last_values.squeeze(-1).cpu().numpy()

        self.buffer.compute_gae(
            last_values=last_values_np,
            last_dones=self._last_dones,
        )

        stats = {}
        if episode_rewards:
            stats["mean_ep_reward"] = float(np.mean(episode_rewards))
            stats["max_ep_reward"] = float(np.max(episode_rewards))
            stats["min_ep_reward"] = float(np.min(episode_rewards))
            stats["mean_ep_length"] = float(np.mean(episode_lengths))
            stats["num_episodes"] = len(episode_rewards)

        # LSTM state stats
        if self._lstm_state is not None:
            lstm_stats = self.model.get_lstm_stats(self._lstm_state)
            stats.update({f"lstm_{k}": v for k, v in lstm_stats.items()})

        return stats

    def update(self, metrics_callback=None) -> Dict[str, float]:
        """
        Perform PPO update with recurrent sequences.

        CRITICAL: We replay the entire sequence using the initial LSTM state
        saved at the start of the rollout. This ensures LSTM states are
        consistent between collection and update phases.
        """
        self.model.train()
        all_metrics = []

        for epoch in range(self.ppo_epochs):
            if metrics_callback:
                metrics_callback({
                    "type": "phase_change",
                    "phase": "PPO_UPDATE",
                    "phase_epoch": f"{epoch + 1}/{self.ppo_epochs}",
                    "phase_details": f"Updating Actor & Critic networks (Epoch {epoch + 1}/{self.ppo_epochs}, clip={self.clip_range})",
                })

            # For recurrent PPO, we process the ENTIRE rollout as a sequence
            # We replay from the saved initial LSTM state
            epoch_metrics = self._recurrent_update_step()
            all_metrics.append(epoch_metrics)

            if self.target_kl is not None and epoch_metrics.get("approx_kl", 0) > self.target_kl:
                logger.debug(f"Early stopping at KL={epoch_metrics['approx_kl']:.4f}")
                break

        self.update_count += 1

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

    def _recurrent_update_step(self) -> Dict[str, float]:
        """
        Single update step for recurrent PPO.

        Uses the full (T, B, ...) sequence with the initial LSTM state.
        """
        # Get the full sequence batch (T, B, ...)
        batches = list(self.buffer.get_recurrent_batches(normalize_advantages=True))
        if not batches:
            return {}

        batch = batches[0]  # Only one batch for full recurrent sequence

        obs = batch["obs"]           # (T, B, C, H, W)
        actions = batch["actions"]   # (T, B)
        old_log_probs = batch["old_log_probs"]  # (T, B)
        advantages = batch["advantages"]         # (T, B)
        returns = batch["returns"]               # (T, B)
        old_values = batch["old_values"]         # (T, B)
        dones = batch["dones"]                   # (T, B)

        T, B = obs.shape[:2]

        # Replay from initial LSTM state
        init_h = self._rollout_initial_lstm_state[0].to(self.device)
        init_c = self._rollout_initial_lstm_state[1].to(self.device)
        init_state: LSTMState = (init_h, init_c)

        # Forward pass through the full sequence
        logits, new_values, _, _ = self.model.forward_sequence(obs, dones, init_state)
        # logits: (T, B, num_actions), new_values: (T, B, 1)

        # Compute new log-probs
        # Flatten for distribution computation
        logits_flat = logits.reshape(T * B, -1)
        actions_flat = actions.reshape(T * B)
        old_log_probs_flat = old_log_probs.reshape(T * B)
        advantages_flat = advantages.reshape(T * B)
        returns_flat = returns.reshape(T * B)
        old_values_flat = old_values.reshape(T * B)
        new_values_flat = new_values.reshape(T * B)

        dist = torch.distributions.Categorical(logits=logits_flat)
        new_log_probs_flat = dist.log_prob(actions_flat)
        entropy = dist.entropy()

        # PPO ratio
        log_ratio = new_log_probs_flat - old_log_probs_flat
        ratio = torch.exp(log_ratio)

        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()

        # Clipped surrogate
        policy_loss_1 = -advantages_flat * ratio
        policy_loss_2 = -advantages_flat * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range
        )
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # Value loss
        value_loss = 0.5 * ((new_values_flat - returns_flat) ** 2).mean()

        # Entropy
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
        """Main training loop for recurrent PPO."""
        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        logger.info("Resetting environments...")
        self._last_obs, _ = self.env.reset()
        self._last_obs = np.array(self._last_obs)
        self._last_dones = np.zeros(self.num_envs, dtype=np.float32)

        # Initialize LSTM state
        self._lstm_state = self.model.initial_state(self.num_envs, self.device)

        self._start_time = time.time()
        last_checkpoint_step = self.global_step

        logger.info(
            f"Starting Recurrent PPO training: total_timesteps={total_timesteps}, "
            f"num_envs={self.num_envs}, rollout_steps={self.rollout_steps}, "
            f"lstm_hidden={self.lstm_hidden}"
        )

        while self.global_step < total_timesteps:
            rollout_stats = self.collect_rollout(metrics_callback=metrics_callback)
            update_metrics = self.update(metrics_callback=metrics_callback)
            update_metrics.update(rollout_stats)

            elapsed = time.time() - self._start_time
            fps = self.global_step / (elapsed + 1e-8)
            update_metrics["fps"] = fps
            update_metrics["elapsed_seconds"] = elapsed

            if self.update_count % 10 == 0:
                logger.info(
                    f"[Recurrent] Step {self.global_step}/{total_timesteps} | "
                    f"Episodes: {self.episode_count} | "
                    f"Mean reward: {rollout_stats.get('mean_ep_reward', 0):.2f} | "
                    f"KL: {update_metrics.get('approx_kl', 0):.4f} | "
                    f"FPS: {fps:.0f}"
                )

            if metrics_callback:
                metrics_callback({"type": "update", **update_metrics})

            if self.global_step - last_checkpoint_step >= checkpoint_interval:
                self.save_checkpoint(tag="latest")
                last_checkpoint_step = self.global_step

        self.save_checkpoint(tag="final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def save_checkpoint(self, tag: str = "latest") -> str:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "update_count": self.update_count,
            "config": self.config,
            "experiment_name": self.experiment_name,
            "model_type": "RecurrentActorCritic",
            "lstm_state_h": self._lstm_state[0].cpu() if self._lstm_state else None,
            "lstm_state_c": self._lstm_state[1].cpu() if self._lstm_state else None,
        }
        path = self.checkpoint_manager.save(checkpoint, tag=tag, step=self.global_step)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
        self.update_count = checkpoint.get("update_count", 0)

        # Restore LSTM state if saved
        if checkpoint.get("lstm_state_h") is not None:
            h = checkpoint["lstm_state_h"].to(self.device)
            c = checkpoint["lstm_state_c"].to(self.device)
            self._lstm_state = (h, c)
        else:
            self._lstm_state = self.model.initial_state(self.num_envs, self.device)

        logger.info(f"Checkpoint loaded: {path} (step={self.global_step})")
