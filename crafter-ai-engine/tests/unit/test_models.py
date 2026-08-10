"""
Unit tests for CNN and Actor-Critic models.
"""

import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from crafter_ai.models.cnn import CrafterCNN
from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic


DEVICE = torch.device("cpu")  # Tests always run on CPU for determinism
BATCH = 4


class TestCrafterCNN:

    def setup_method(self):
        self.cnn = CrafterCNN(in_channels=3, out_features=512)
        self.obs = torch.randn(BATCH, 3, 64, 64)

    def test_output_shape(self):
        out = self.cnn(self.obs)
        assert out.shape == (BATCH, 512)

    def test_output_dtype(self):
        out = self.cnn(self.obs)
        assert out.dtype == torch.float32

    def test_different_input_channels(self):
        cnn = CrafterCNN(in_channels=12, out_features=512)
        obs = torch.randn(BATCH, 12, 64, 64)
        out = cnn(obs)
        assert out.shape == (BATCH, 512)

    def test_custom_out_features(self):
        cnn = CrafterCNN(in_channels=3, out_features=256)
        out = cnn(self.obs)
        assert out.shape == (BATCH, 256)

    def test_sequence_input_5d(self):
        """CNN should handle (B, T, C, H, W) inputs."""
        obs = torch.randn(BATCH, 10, 3, 64, 64)
        out = self.cnn(obs)
        assert out.shape == (BATCH, 10, 512)

    def test_gradients_flow(self):
        obs = self.obs.requires_grad_(True)
        out = self.cnn(obs)
        loss = out.mean()
        loss.backward()
        assert obs.grad is not None
        assert not obs.grad.isnan().any()

    def test_parameters_nonzero(self):
        """After init, CNN should have nonzero weights."""
        for name, param in self.cnn.named_parameters():
            assert not param.data.isnan().any(), f"NaN in {name}"

    def test_flat_size_correct(self):
        """Test that flat_size calculation is correct."""
        flat = self.cnn.get_flat_size()
        assert flat > 0
        # 3 conv layers: 64→31→15→7, so 64 channels * 7 * 7 = 3136
        assert flat == 64 * 7 * 7, f"Expected 3136, got {flat}"


class TestActorCritic:

    def setup_method(self):
        self.model = ActorCritic(in_channels=3, num_actions=17).to(DEVICE)
        self.obs = torch.randn(BATCH, 3, 64, 64)

    def test_forward_shapes(self):
        logits, values, features = self.model(self.obs)
        assert logits.shape == (BATCH, 17)
        assert values.shape == (BATCH, 1)
        assert features.shape == (BATCH, 512)

    def test_get_action_and_value_shapes(self):
        action, log_prob, entropy, value, probs = self.model.get_action_and_value(self.obs)
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert entropy.shape == (BATCH,)
        assert value.shape == (BATCH, 1)
        assert probs.shape == (BATCH, 17)

    def test_action_range(self):
        action, _, _, _, _ = self.model.get_action_and_value(self.obs)
        assert action.min() >= 0
        assert action.max() < 17

    def test_probs_sum_to_one(self):
        _, _, _, _, probs = self.model.get_action_and_value(self.obs)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_given_action_log_prob(self):
        """When action is given, compute its log prob correctly."""
        action = torch.zeros(BATCH, dtype=torch.long)
        _, log_prob, _, _, probs = self.model.get_action_and_value(self.obs, action=action)
        expected = torch.log(probs[:, 0])
        torch.testing.assert_close(log_prob, expected, atol=1e-5, rtol=1e-5)

    def test_entropy_positive(self):
        _, _, entropy, _, _ = self.model.get_action_and_value(self.obs)
        assert (entropy > 0).all(), "Entropy should be positive"

    def test_get_value(self):
        value = self.model.get_value(self.obs)
        assert value.shape == (BATCH, 1)

    def test_total_parameters(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 100_000, "Model should have substantial parameters"

    def test_gradients_propagate(self):
        action, log_prob, entropy, value, _ = self.model.get_action_and_value(self.obs)
        loss = -log_prob.mean() + value.mean() - entropy.mean()
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


class TestRecurrentActorCritic:

    def setup_method(self):
        self.model = RecurrentActorCritic(
            in_channels=3, num_actions=17, lstm_hidden=256
        ).to(DEVICE)
        self.obs = torch.randn(BATCH, 3, 64, 64)
        self.done = torch.zeros(BATCH)
        self.state = self.model.initial_state(BATCH, DEVICE)

    def test_initial_state_shape(self):
        h, c = self.state
        assert h.shape == (1, BATCH, 256)
        assert c.shape == (1, BATCH, 256)

    def test_initial_state_zeros(self):
        h, c = self.state
        assert h.sum().item() == 0.0
        assert c.sum().item() == 0.0

    def test_get_action_and_value_shapes(self):
        action, log_prob, entropy, value, probs, new_state = self.model.get_action_and_value(
            self.obs, self.done, self.state
        )
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert entropy.shape == (BATCH,)
        assert value.shape == (BATCH, 1)
        assert probs.shape == (BATCH, 17)
        h, c = new_state
        assert h.shape == (1, BATCH, 256)
        assert c.shape == (1, BATCH, 256)

    def test_lstm_state_updated(self):
        """LSTM state should change after a step."""
        _, _, _, _, _, new_state = self.model.get_action_and_value(
            self.obs, self.done, self.state
        )
        h_initial, _ = self.state
        h_new, _ = new_state
        # With non-trivial input, state should change
        assert not torch.allclose(h_initial, h_new), "LSTM state should update"

    def test_lstm_state_reset_on_done(self):
        """When done=True, LSTM state for that env should be zeroed before next step."""
        # Run one step to get non-zero state
        _, _, _, _, _, state_after1 = self.model.get_action_and_value(
            self.obs, self.done, self.state
        )

        # Now run with done=True for all envs
        done_all = torch.ones(BATCH)
        obs2 = torch.randn(BATCH, 3, 64, 64)

        # In forward_sequence, done at t=0 would reset at t=1
        # We test forward_sequence directly
        T = 3
        obs_seq = torch.randn(T, BATCH, 3, 64, 64)
        # Mark step 1 as done (will reset at step 2)
        done_seq = torch.zeros(T, BATCH)
        done_seq[1, :] = 1.0  # All envs done at step 1

        logits, values, lstm_out, final_state = self.model.forward_sequence(
            obs_seq, done_seq, self.state
        )
        assert logits.shape == (T, BATCH, 17)
        assert values.shape == (T, BATCH, 1)

    def test_probs_sum_to_one(self):
        _, _, _, _, probs, _ = self.model.get_action_and_value(
            self.obs, self.done, self.state
        )
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=1e-5)

    def test_forward_sequence_shapes(self):
        T = 10
        obs_seq = torch.randn(T, BATCH, 3, 64, 64)
        done_seq = torch.zeros(T, BATCH)
        logits, values, lstm_out, final_state = self.model.forward_sequence(
            obs_seq, done_seq, self.state
        )
        assert logits.shape == (T, BATCH, 17)
        assert values.shape == (T, BATCH, 1)
        assert lstm_out.shape == (T, BATCH, 256)

    def test_lstm_stats(self):
        stats = self.model.get_lstm_stats(self.state)
        assert "h_norm" in stats
        assert "c_norm" in stats

    def test_total_parameters(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 200_000

    def test_temporal_state_persistence(self):
        """Hidden state at step t should influence output at step t+1."""
        obs1 = torch.randn(BATCH, 3, 64, 64)
        obs2 = torch.ones(BATCH, 3, 64, 64)  # constant obs for repeatability

        # Run two steps, maintaining state
        _, _, _, _, _, state1 = self.model.get_action_and_value(obs1, self.done, self.state)
        _, _, _, _, probs_a, _ = self.model.get_action_and_value(obs2, self.done, state1)

        # Run one step fresh with same obs2 from zero state
        _, _, _, _, probs_b, _ = self.model.get_action_and_value(obs2, self.done, self.state)

        # Probs should differ because LSTM has different hidden state
        assert not torch.allclose(probs_a, probs_b, atol=1e-5), (
            "LSTM temporal memory should affect outputs"
        )

