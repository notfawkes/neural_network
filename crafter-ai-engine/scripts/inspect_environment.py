#!/usr/bin/env python3
"""
inspect_environment.py
======================
Inspect the Crafter environment, verify API, and report hardware capabilities.

Usage:
    python scripts/inspect_environment.py
"""

import sys
import os
import time
import numpy as np

# Add engine root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch

print("=" * 70)
print("CRAfter AI — Environment Inspection")
print("=" * 70)

# ─── Python & torch ───────────────────────────────────────────────────────────
print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

device_name = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
print(f"Selected device: {device}")

# ─── Crafter Environment ──────────────────────────────────────────────────────
print("\n--- Crafter Environment ---")
from crafter_ai.env.adapter import CrafterEnv, VecCrafterEnv, ACTION_NAMES, ACHIEVEMENT_NAMES

env = CrafterEnv(reward=True, seed=42)
obs, info = env.reset()

print(f"Observation shape (CHW): {obs.shape}")
print(f"Observation dtype:       {obs.dtype}")
print(f"Observation range:       [{obs.min():.3f}, {obs.max():.3f}]")
print(f"Number of actions:       {env.num_actions}")
print(f"Number of achievements:  {len(ACHIEVEMENT_NAMES)}")

print(f"\nAction names:")
for i, name in enumerate(ACTION_NAMES):
    print(f"  {i:2d}: {name}")

print(f"\nAchievement names:")
for i, name in enumerate(ACHIEVEMENT_NAMES):
    print(f"  {i:2d}: {name}")

# ─── Step test ────────────────────────────────────────────────────────────────
print("\n--- Step test ---")
obs2, reward, terminated, truncated, info = env.step(1)  # move_left
print(f"Step obs shape: {obs2.shape}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}, Truncated: {truncated}")
print(f"Info keys: {list(info.keys())}")
print(f"Player position: {info.get('player_pos')}")
print(f"Episode step: {info.get('episode_step')}")

# ─── Vectorized env ───────────────────────────────────────────────────────────
print("\n--- Vectorized Environment (4 envs) ---")
vec_env = VecCrafterEnv(num_envs=4, reward=True)
obs_batch, infos = vec_env.reset()
print(f"Batch obs shape: {obs_batch.shape}")

actions = np.array([0, 1, 2, 3])
obs_batch2, rewards, terminated, truncated, infos = vec_env.step(actions)
print(f"Batch step obs shape: {obs_batch2.shape}")
print(f"Rewards: {rewards}")
print(f"Terminated: {terminated}, Truncated: {truncated}")

# ─── CNN test ─────────────────────────────────────────────────────────────────
print("\n--- CNN Feature Extractor ---")
from crafter_ai.models.cnn import CrafterCNN

cnn = CrafterCNN(in_channels=3, out_features=512).to(device)
test_obs = torch.as_tensor(obs_batch, dtype=torch.float32).to(device)
with torch.no_grad():
    features = cnn(test_obs)
print(f"CNN input shape:  {test_obs.shape}")
print(f"CNN output shape: {features.shape}")
print(f"CNN parameters:   {sum(p.numel() for p in cnn.parameters()):,}")

# ─── Actor-Critic test ────────────────────────────────────────────────────────
print("\n--- ActorCritic (MLP) ---")
from crafter_ai.models.actor_critic import ActorCritic

ac = ActorCritic(in_channels=3, num_actions=17).to(device)
with torch.no_grad():
    action, log_prob, entropy, value, probs = ac.get_action_and_value(test_obs)
print(f"Actions shape:    {action.shape}")
print(f"Log probs shape:  {log_prob.shape}")
print(f"Value shape:      {value.shape}")
print(f"Probs shape:      {probs.shape}")
print(f"Probs sum:        {probs.sum(dim=-1)}")  # Should be ~1.0
print(f"AC parameters:    {sum(p.numel() for p in ac.parameters()):,}")

# ─── Recurrent Actor-Critic test ─────────────────────────────────────────────
print("\n--- RecurrentActorCritic (CNN + LSTM) ---")
from crafter_ai.models.actor_critic import RecurrentActorCritic

rac = RecurrentActorCritic(in_channels=3, num_actions=17, lstm_hidden=256).to(device)
lstm_state = rac.initial_state(4, device)
done_tensor = torch.zeros(4, device=device)
with torch.no_grad():
    action_r, log_prob_r, entropy_r, value_r, probs_r, new_state = rac.get_action_and_value(
        test_obs, done_tensor, lstm_state
    )
print(f"Actions shape:    {action_r.shape}")
print(f"Value shape:      {value_r.shape}")
print(f"LSTM h shape:     {new_state[0].shape}")
print(f"LSTM c shape:     {new_state[1].shape}")
print(f"RAC parameters:   {sum(p.numel() for p in rac.parameters()):,}")

# ─── Throughput benchmark ─────────────────────────────────────────────────────
print("\n--- Throughput Benchmark ---")
from crafter_ai.utils.device import benchmark_device

result = benchmark_device(device, batch_size=4)
print(f"Device: {result['device']}")
print(f"Steps/sec estimate: {result['steps_per_second_estimate']:.0f}")
print(f"Time per step: {result['time_per_step_ms']:.2f} ms")

print("\n" + "=" * 70)
print("✓ All checks passed. Environment ready for training.")
print("=" * 70)

