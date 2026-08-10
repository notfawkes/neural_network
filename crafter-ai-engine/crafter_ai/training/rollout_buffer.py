"""
Rollout Buffer
==============

Stores experience collected from N parallel environments over T timesteps.

For MLP PPO (ActorCritic):
  - Stores flat transitions: obs, actions, rewards, dones, log_probs, values
  - Computes advantages using GAE
  - Yields mini-batches (can shuffle individual transitions)

For Recurrent PPO (RecurrentActorCritic):
  - Preserves temporal order within each environment
  - Yields WHOLE environment sequences for LSTM processing
  - Never shuffles across time within a sequence
"""

import torch
import numpy as np
from typing import Generator, Optional, Tuple, List


class RolloutBuffer:
    """
    Rollout buffer for PPO training.

    Collects T steps from B environments. Total transitions: T * B.

    Parameters
    ----------
    rollout_steps : int
        Number of steps T collected per environment.
    num_envs : int
        Number of parallel environments B.
    obs_shape : tuple
        Shape of a single observation (C, H, W).
    num_actions : int
        Size of the action space.
    device : torch.device
        Device for returned tensors.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self._pos = 0
        self._full = False

        # Pre-allocate buffers on CPU for efficiency (move to device in mini-batches)
        T, B = rollout_steps, num_envs

        self.obs = np.zeros((T, B, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((T, B), dtype=np.int64)
        self.rewards = np.zeros((T, B), dtype=np.float32)
        self.dones = np.zeros((T, B), dtype=np.float32)  # 1.0 if done
        self.log_probs = np.zeros((T, B), dtype=np.float32)
        self.values = np.zeros((T, B), dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros((T, B), dtype=np.float32)
        self.returns = np.zeros((T, B), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,         # (B, C, H, W)
        actions: np.ndarray,     # (B,)
        rewards: np.ndarray,     # (B,)
        dones: np.ndarray,       # (B,)
        log_probs: np.ndarray,   # (B,)
        values: np.ndarray,      # (B,)
    ):
        """Add a single timestep of experience."""
        t = self._pos
        self.obs[t] = obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.log_probs[t] = log_probs
        self.values[t] = values
        self._pos += 1
        if self._pos >= self.rollout_steps:
            self._full = True

    def compute_gae(
        self,
        last_values: np.ndarray,  # (B,)
        last_dones: np.ndarray,   # (B,)
    ):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        GAE formula:
          δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t = δ_t + (γ * λ) * (1 - done_t) * A_{t+1}

        Parameters
        ----------
        last_values : (B,) — critic values for the state AFTER the rollout
        last_dones : (B,) — whether the last state was terminal
        """
        T = self.rollout_steps
        last_gae_lam = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def reset(self):
        """Clear the buffer for the next rollout."""
        self._pos = 0
        self._full = False

    def is_full(self) -> bool:
        return self._full

    def get_flat_batches(
        self, batch_size: int, normalize_advantages: bool = True
    ) -> Generator:
        """
        Yield randomized mini-batches for MLP PPO.

        Flattens T * B transitions and shuffles. Maintains the flat batch
        assumption (no temporal dependencies for MLP).

        Yields
        ------
        dict with keys: obs, actions, log_probs, advantages, returns, values
        Each value is a tensor on self.device.
        """
        total = self.rollout_steps * self.num_envs
        indices = np.random.permutation(total)

        obs_flat = self.obs.reshape(total, *self.obs_shape)
        actions_flat = self.actions.reshape(total)
        log_probs_flat = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat = self.returns.reshape(total)
        values_flat = self.values.reshape(total)

        if normalize_advantages:
            adv_mean = advantages_flat.mean()
            adv_std = advantages_flat.std() + 1e-8
            advantages_flat = (advantages_flat - adv_mean) / adv_std

        for start in range(0, total, batch_size):
            end = start + batch_size
            if end > total:
                break
            batch_idx = indices[start:end]
            yield {
                "obs": torch.as_tensor(obs_flat[batch_idx], dtype=torch.float32).to(self.device),
                "actions": torch.as_tensor(actions_flat[batch_idx], dtype=torch.long).to(self.device),
                "old_log_probs": torch.as_tensor(log_probs_flat[batch_idx], dtype=torch.float32).to(self.device),
                "advantages": torch.as_tensor(advantages_flat[batch_idx], dtype=torch.float32).to(self.device),
                "returns": torch.as_tensor(returns_flat[batch_idx], dtype=torch.float32).to(self.device),
                "old_values": torch.as_tensor(values_flat[batch_idx], dtype=torch.float32).to(self.device),
            }

    def get_recurrent_batches(
        self,
        normalize_advantages: bool = True,
    ) -> Generator:
        """
        Yield full-sequence batches for Recurrent PPO.

        CRITICAL: Temporal order is preserved within each environment's sequence.
        We yield full (T, 1, ...) sequences, one environment at a time, then
        can combine multiple envs into (T, mini_B, ...) batches.

        For correctness with LSTM:
        - Each environment's T-step sequence is kept intact.
        - We shuffle the ORDER of environments, not the timesteps.

        Yields
        ------
        dict with keys: obs, actions, log_probs, advantages, returns, values, dones
        Each value has shape (T, B_mini, ...) as tensors on self.device.
        """
        T = self.rollout_steps
        B = self.num_envs
        env_indices = np.random.permutation(B)

        adv = self.advantages  # (T, B)
        if normalize_advantages:
            adv_mean = adv.mean()
            adv_std = adv.std() + 1e-8
            adv = (adv - adv_mean) / adv_std

        # Each env's sequence is one "batch" for recurrent PPO
        # We can optionally combine multiple envs: here we yield all envs together
        # (T, B, ...) keeping temporal order
        # Reorder envs randomly (across-env, not across-time)
        yield {
            "obs": torch.as_tensor(
                self.obs[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B, C, H, W)
            "actions": torch.as_tensor(
                self.actions[:, env_indices], dtype=torch.long
            ).to(self.device),  # (T, B)
            "old_log_probs": torch.as_tensor(
                self.log_probs[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B)
            "advantages": torch.as_tensor(
                adv[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B)
            "returns": torch.as_tensor(
                self.returns[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B)
            "old_values": torch.as_tensor(
                self.values[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B)
            "dones": torch.as_tensor(
                self.dones[:, env_indices], dtype=torch.float32
            ).to(self.device),  # (T, B)
        }

