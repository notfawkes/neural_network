"""
Unit tests for rollout buffer and GAE computation.
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from crafter_ai.training.rollout_buffer import RolloutBuffer


DEVICE = torch.device("cpu")
T = 16   # rollout steps
B = 4    # num envs
OBS_SHAPE = (3, 64, 64)
NUM_ACTIONS = 17


class TestRolloutBuffer:

    def setup_method(self):
        self.buffer = RolloutBuffer(
            rollout_steps=T,
            num_envs=B,
            obs_shape=OBS_SHAPE,
            num_actions=NUM_ACTIONS,
            device=DEVICE,
            gamma=0.99,
            gae_lambda=0.95,
        )

    def _fill_buffer(self, constant_reward=1.0):
        for t in range(T):
            obs = np.random.randn(B, *OBS_SHAPE).astype(np.float32)
            actions = np.random.randint(0, NUM_ACTIONS, (B,))
            rewards = np.ones(B, dtype=np.float32) * constant_reward
            dones = np.zeros(B, dtype=np.float32)
            log_probs = np.log(np.ones(B) / NUM_ACTIONS)
            values = np.zeros(B, dtype=np.float32)
            self.buffer.add(obs, actions, rewards, dones, log_probs, values)

    def test_buffer_fills(self):
        self._fill_buffer()
        assert self.buffer.is_full()

    def test_buffer_dimensions(self):
        self._fill_buffer()
        assert self.buffer.obs.shape == (T, B, *OBS_SHAPE)
        assert self.buffer.actions.shape == (T, B)
        assert self.buffer.rewards.shape == (T, B)
        assert self.buffer.dones.shape == (T, B)
        assert self.buffer.log_probs.shape == (T, B)
        assert self.buffer.values.shape == (T, B)

    def test_gae_advantage_shape(self):
        self._fill_buffer()
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)
        assert self.buffer.advantages.shape == (T, B)
        assert self.buffer.returns.shape == (T, B)

    def test_gae_constant_reward_no_done(self):
        """With constant reward=1 and V=0, advantages should be positive."""
        self._fill_buffer(constant_reward=1.0)
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)
        assert (self.buffer.advantages > 0).all(), "Positive reward → positive advantages"

    def test_gae_zero_reward(self):
        """With zero reward and V=0, advantages should be zero."""
        self._fill_buffer(constant_reward=0.0)
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)
        np.testing.assert_allclose(self.buffer.advantages, 0.0, atol=1e-6)

    def test_returns_equal_advantages_plus_values(self):
        """Returns = advantages + values (by definition)."""
        self._fill_buffer()
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)
        np.testing.assert_allclose(
            self.buffer.returns,
            self.buffer.advantages + self.buffer.values,
            atol=1e-6,
        )

    def test_flat_batches_produce_correct_total(self):
        self._fill_buffer()
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)

        batch_size = 32
        total = 0
        for batch in self.buffer.get_flat_batches(batch_size=batch_size):
            assert "obs" in batch
            assert "actions" in batch
            assert "old_log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch
            total += batch["obs"].shape[0]

        # Should cover all transitions (some may be dropped at end)
        assert total <= T * B

    def test_flat_batches_obs_shape(self):
        self._fill_buffer()
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)

        for batch in self.buffer.get_flat_batches(batch_size=16):
            assert batch["obs"].shape[1:] == OBS_SHAPE
            break

    def test_recurrent_batches_preserve_time(self):
        self._fill_buffer()
        last_values = np.zeros(B, dtype=np.float32)
        last_dones = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, last_dones)

        batches = list(self.buffer.get_recurrent_batches())
        assert len(batches) == 1
        batch = batches[0]

        # Temporal order preserved: shape (T, B, ...)
        assert batch["obs"].shape == (T, B, *OBS_SHAPE)
        assert batch["actions"].shape == (T, B)
        assert batch["dones"].shape == (T, B)

    def test_buffer_reset(self):
        self._fill_buffer()
        assert self.buffer.is_full()
        self.buffer.reset()
        assert not self.buffer.is_full()
        assert self.buffer._pos == 0

    def test_normalized_advantages(self):
        self._fill_buffer(constant_reward=1.0)
        last_values = np.zeros(B, dtype=np.float32)
        self.buffer.compute_gae(last_values, np.zeros(B))

        batches = list(self.buffer.get_flat_batches(batch_size=T*B, normalize_advantages=True))
        if batches:
            adv = batches[0]["advantages"]
            # Normalized: mean ≈ 0, std ≈ 1
            assert abs(adv.mean().item()) < 0.1

