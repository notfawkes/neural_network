#!/usr/bin/env python3
"""
compare.py
==========
Compare two model checkpoints on identical observations.

Usage:
    python scripts/compare.py --checkpoint-a A.pt --checkpoint-b B.pt
    python scripts/compare.py --checkpoint-a A.pt --checkpoint-b B.pt --obs-source recordings/ep_00000001.npz
"""

import sys
import os
import argparse
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import torch.nn.functional as F

from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
from crafter_ai.env.adapter import CrafterEnv, ACTION_NAMES
from crafter_ai.utils.device import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compare")


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint.get("model_type", "ActorCritic")
    config = checkpoint.get("config", {})

    if model_type == "RecurrentActorCritic":
        model = RecurrentActorCritic(
            in_channels=3, num_actions=17,
            cnn_out=config.get("cnn_out_features", 512),
            lstm_hidden=config.get("lstm_hidden_size", 256),
        ).to(device)
    else:
        model = ActorCritic(
            in_channels=3, num_actions=17,
            cnn_out=config.get("cnn_out_features", 512),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_type, checkpoint


def get_probs(model, obs_tensor: torch.Tensor, model_type: str, device: torch.device) -> np.ndarray:
    """Get action probabilities for a batch of observations."""
    with torch.no_grad():
        if model_type == "RecurrentActorCritic":
            state = model.initial_state(obs_tensor.shape[0], device)
            done = torch.zeros(obs_tensor.shape[0], device=device)
            _, _, _, _, probs, _ = model.get_action_and_value(obs_tensor, done, state)
        else:
            _, _, _, _, probs = model.get_action_and_value(obs_tensor)
    return probs.cpu().numpy()


def sample_observations(source: str = None, n: int = 4) -> np.ndarray:
    """Get sample observations for comparison."""
    if source and os.path.exists(source):
        data = np.load(source)
        if "obs_frames" in data:
            frames = data["obs_frames"]
            indices = np.linspace(0, len(frames) - 1, n, dtype=int)
            obs_hwc = frames[indices]  # (n, 64, 64, 3) uint8
            from crafter_ai.env.adapter import CrafterEnv
            return CrafterEnv.preprocess_batch(obs_hwc)  # (n, 3, 64, 64)
    
    # Generate random observations using the environment
    logger.info("No obs source provided, using fresh environment observations")
    env = CrafterEnv(reward=True, seed=42)
    obs_list = []
    obs, _ = env.reset()
    obs_list.append(obs)
    for _ in range(n - 1):
        for _ in range(50):  # Step around
            obs, _, terminated, truncated, _ = env.step(np.random.randint(0, 17))
            if terminated or truncated:
                obs, _ = env.reset()
        obs_list.append(obs)
    return np.stack(obs_list, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--obs-source", default=None, help="NPZ file with obs_frames")
    parser.add_argument("--n-obs", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Load models
    logger.info(f"Loading Model A: {args.checkpoint_a}")
    model_a, type_a, ckpt_a = load_model(args.checkpoint_a, device)

    logger.info(f"Loading Model B: {args.checkpoint_b}")
    model_b, type_b, ckpt_b = load_model(args.checkpoint_b, device)

    # Get observations
    obs_np = sample_observations(args.obs_source, args.n_obs)
    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32).to(device)

    # Get probabilities
    probs_a = get_probs(model_a, obs_tensor, type_a, device)
    probs_b = get_probs(model_b, obs_tensor, type_b, device)

    # Print comparison
    print("\n" + "=" * 80)
    print(f"CHECKPOINT COMPARISON")
    print(f"Model A: {os.path.basename(args.checkpoint_a)} (step {ckpt_a.get('global_step', '?')})")
    print(f"Model B: {os.path.basename(args.checkpoint_b)} (step {ckpt_b.get('global_step', '?')})")
    print("=" * 80)

    for obs_i in range(args.n_obs):
        print(f"\n--- Observation {obs_i + 1} ---")
        pa = probs_a[obs_i]
        pb = probs_b[obs_i]

        print(f"{'Action':<25} {'Model A':>8} {'Model B':>8} {'Δ':>8}")
        print("-" * 55)

        sorted_by_a = sorted(range(17), key=lambda i: pa[i], reverse=True)
        for i in sorted_by_a[:8]:  # Top 8 actions
            delta = pb[i] - pa[i]
            arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else " ")
            print(f"  {ACTION_NAMES[i]:<23} {pa[i]:7.1%}  {pb[i]:7.1%}  {arrow}{abs(delta):6.1%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

