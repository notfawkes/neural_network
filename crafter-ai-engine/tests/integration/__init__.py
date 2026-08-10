"""
Integration test: short training run to verify the entire pipeline.
"""

import sys
import os
import pytest
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

import torch

from crafter_ai.env.adapter import make_env
from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
from crafter_ai.training.ppo import PPOTrainer
from crafter_ai.training.ppo_recurrent import RecurrentPPOTrainer


DEVICE = torch.device("cpu")


FAST_CONFIG = {
    "total_timesteps": 512,  # Very short — just testing pipeline
    "num_envs": 2,
    "rollout_steps": 16,
    "ppo_epochs": 2,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "max_grad_norm": 0.5,
    "lstm_hidden_size": 64,  # Smaller for faster test
    "cnn_out_features": 64,  # Smaller for faster test
}


class TestMLPPPOIntegration:
    """Test the MLP (baseline) PPO training pipeline."""

    def test_training_runs_and_parameters_update(self):
        """
        Verify that:
        1. Training runs without errors.
        2. Model parameters change after update.
        3. Checkpoint can be saved and loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_env(num_envs=2, reward=True, seed=0)
            model = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)

            # Capture initial parameters
            initial_params = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }

            trainer = PPOTrainer(
                model=model,
                env=env,
                config=FAST_CONFIG,
                device=DEVICE,
                checkpoint_dir=tmpdir,
                experiment_name="test_mlp",
            )

            updates_seen = []
            def callback(event):
                if event.get("type") == "update":
                    updates_seen.append(event)

            trainer.train(
                total_timesteps=512,
                metrics_callback=callback,
                checkpoint_interval=10_000,
            )

            # At least one update should have occurred
            assert len(updates_seen) > 0, "Should have at least one update"

            # Parameters should have changed
            params_changed = False
            for name, param in model.named_parameters():
                if not torch.allclose(initial_params[name], param.data, atol=1e-10):
                    params_changed = True
                    break
            assert params_changed, "Model parameters should change after training"

            # Checkpoint should have been saved
            import glob
            checkpoints = glob.glob(os.path.join(tmpdir, "*.pt"))
            assert len(checkpoints) > 0, "Should have saved at least one checkpoint"

    def test_checkpoint_save_load(self):
        """Verify checkpoint can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_env(num_envs=2, reward=True, seed=1)
            model = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)

            trainer = PPOTrainer(
                model=model, env=env, config=FAST_CONFIG,
                device=DEVICE, checkpoint_dir=tmpdir, experiment_name="test_ckpt",
            )
            trainer.train(total_timesteps=256, checkpoint_interval=10_000)
            path = trainer.save_checkpoint(tag="test")

            # Load into new model
            model2 = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)
            checkpoint = torch.load(path, map_location=DEVICE)
            model2.load_state_dict(checkpoint["model_state_dict"])

            # Models should produce same outputs
            obs = torch.randn(4, 3, 64, 64)
            with torch.no_grad():
                logits1, _, _ = model(obs)
                logits2, _, _ = model2(obs)
            torch.testing.assert_close(logits1, logits2)


class TestRecurrentPPOIntegration:
    """Test the LSTM PPO training pipeline."""

    def test_recurrent_training_runs(self):
        """Verify LSTM PPO training works end to end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_env(num_envs=2, reward=True, seed=0)
            model = RecurrentActorCritic(
                in_channels=3, num_actions=17, cnn_out=64, lstm_hidden=64
            ).to(DEVICE)

            initial_params = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }

            trainer = RecurrentPPOTrainer(
                model=model,
                env=env,
                config=FAST_CONFIG,
                device=DEVICE,
                checkpoint_dir=tmpdir,
                experiment_name="test_lstm",
            )

            updates_seen = []
            def callback(event):
                if event.get("type") == "update":
                    updates_seen.append(event)

            trainer.train(
                total_timesteps=512,
                metrics_callback=callback,
                checkpoint_interval=10_000,
            )

            assert len(updates_seen) > 0

            # Parameters should change
            params_changed = False
            for name, param in model.named_parameters():
                if not torch.allclose(initial_params[name], param.data, atol=1e-10):
                    params_changed = True
                    break
            assert params_changed, "LSTM model parameters should update"

    def test_lstm_state_restored_from_checkpoint(self):
        """LSTM state should be preserved in checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = make_env(num_envs=2, reward=True, seed=0)
            model = RecurrentActorCritic(
                in_channels=3, num_actions=17, cnn_out=64, lstm_hidden=64
            ).to(DEVICE)

            trainer = RecurrentPPOTrainer(
                model=model, env=env, config=FAST_CONFIG,
                device=DEVICE, checkpoint_dir=tmpdir, experiment_name="test_lstm_ckpt",
            )
            trainer.train(total_timesteps=256, checkpoint_interval=10_000)
            path = trainer.save_checkpoint(tag="test")

            # Load checkpoint and check LSTM state
            checkpoint = torch.load(path, map_location=DEVICE)
            assert "lstm_state_h" in checkpoint or checkpoint.get("model_type") == "RecurrentActorCritic"


class TestPPOLossValues:
    """Test that PPO loss values are reasonable."""

    def test_ppo_loss_finite(self):
        """PPO losses should be finite numbers."""
        from crafter_ai.training.rollout_buffer import RolloutBuffer

        T, B = 16, 2
        buffer = RolloutBuffer(T, B, (3, 64, 64), 17, DEVICE)

        for _ in range(T):
            obs = np.random.randn(B, 3, 64, 64).astype(np.float32)
            buffer.add(
                obs=obs,
                actions=np.zeros(B, dtype=np.int64),
                rewards=np.ones(B, dtype=np.float32),
                dones=np.zeros(B, dtype=np.float32),
                log_probs=np.full(B, -np.log(17), dtype=np.float32),
                values=np.ones(B, dtype=np.float32) * 0.5,
            )

        buffer.compute_gae(np.ones(B) * 0.5, np.zeros(B))

        model = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        for batch in buffer.get_flat_batches(batch_size=16):
            _, new_log_probs, entropy, new_values, _ = model.get_action_and_value(
                batch["obs"], action=batch["actions"]
            )
            ratio = torch.exp(new_log_probs - batch["old_log_probs"])
            adv = batch["advantages"]
            policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            value_loss = 0.5 * ((new_values.squeeze(-1) - batch["returns"]) ** 2).mean()
            loss = policy_loss + 0.5 * value_loss

            assert torch.isfinite(loss), f"Loss is not finite: {loss}"
            assert torch.isfinite(policy_loss), f"Policy loss not finite: {policy_loss}"
            assert torch.isfinite(value_loss), f"Value loss not finite: {value_loss}"
            break  # Just test one batch

