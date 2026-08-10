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
    "total_timesteps": 512,
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
    "lstm_hidden_size": 64,
    "cnn_out_features": 64,
}


def test_mlp_ppo_training_updates_params():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = make_env(num_envs=2, reward=True, seed=0)
        model = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)
        initial_params = {n: p.data.clone() for n, p in model.named_parameters()}

        trainer = PPOTrainer(
            model=model, env=env, config=FAST_CONFIG,
            device=DEVICE, checkpoint_dir=tmpdir, experiment_name="test_mlp",
        )

        updates = []
        trainer.train(total_timesteps=512, metrics_callback=lambda e: updates.append(e) if e.get("type") == "update" else None, checkpoint_interval=100_000)

        assert len(updates) > 0
        params_changed = any(
            not torch.allclose(initial_params[n], p.data, atol=1e-10)
            for n, p in model.named_parameters()
        )
        assert params_changed


def test_recurrent_ppo_training_updates_params():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = make_env(num_envs=2, reward=True, seed=0)
        model = RecurrentActorCritic(in_channels=3, num_actions=17, cnn_out=64, lstm_hidden=64).to(DEVICE)
        initial_params = {n: p.data.clone() for n, p in model.named_parameters()}

        trainer = RecurrentPPOTrainer(
            model=model, env=env, config=FAST_CONFIG,
            device=DEVICE, checkpoint_dir=tmpdir, experiment_name="test_lstm",
        )

        updates = []
        trainer.train(total_timesteps=512, metrics_callback=lambda e: updates.append(e) if e.get("type") == "update" else None, checkpoint_interval=100_000)

        assert len(updates) > 0
        params_changed = any(
            not torch.allclose(initial_params[n], p.data, atol=1e-10)
            for n, p in model.named_parameters()
        )
        assert params_changed


def test_checkpoint_save_load_mlp():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = make_env(num_envs=2, reward=True, seed=1)
        model = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)

        trainer = PPOTrainer(
            model=model, env=env, config=FAST_CONFIG,
            device=DEVICE, checkpoint_dir=tmpdir, experiment_name="test_ckpt",
        )
        trainer.train(total_timesteps=256, checkpoint_interval=100_000)
        path = trainer.save_checkpoint(tag="test")

        model2 = ActorCritic(in_channels=3, num_actions=17, cnn_out=64).to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE)
        model2.load_state_dict(checkpoint["model_state_dict"])

        obs = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            logits1, _, _ = model(obs)
            logits2, _, _ = model2(obs)
        torch.testing.assert_close(logits1, logits2)
