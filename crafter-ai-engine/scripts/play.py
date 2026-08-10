#!/usr/bin/env python3
"""
play.py
=======
Watch a trained agent play Crafter interactively.

Usage:
    python scripts/play.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt
    python scripts/play.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt --episodes 5
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import numpy as np

from crafter_ai.env.adapter import CrafterEnv, ACTION_NAMES, ACHIEVEMENT_NAMES
from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
from crafter_ai.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps (seconds)")
    parser.add_argument("--greedy", action="store_true", help="Use greedy (argmax) instead of sampling")
    args = parser.parse_args()

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
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

    print(f"Loaded {model_type} from: {args.checkpoint}")
    print(f"Checkpoint step: {checkpoint.get('global_step', 0)}")
    print(f"Playing {args.episodes} episodes...\n")

    env = CrafterEnv(reward=True, seed=42)
    is_recurrent = isinstance(model, RecurrentActorCritic)

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        unlocked = set()

        if is_recurrent:
            lstm_state = model.initial_state(1, device)
            done_tensor = torch.zeros(1, device=device)

        print(f"\n--- Episode {episode + 1} ---")

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                if is_recurrent:
                    action, _, _, _, probs, lstm_state = model.get_action_and_value(
                        obs_t, done_tensor, lstm_state
                    )
                    if args.greedy:
                        action = probs.argmax(dim=-1)
                else:
                    action, _, _, _, probs = model.get_action_and_value(obs_t)
                    if args.greedy:
                        action = probs.argmax(dim=-1)

            action_int = int(action.item())
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Print step info
            if step % 50 == 1 or done:
                top3_probs = sorted(
                    [(p, ACTION_NAMES[i]) for i, p in enumerate(probs[0].cpu().numpy())],
                    reverse=True
                )[:3]
                top3_str = ", ".join([f"{n}:{p:.2f}" for p, n in top3_probs])
                print(
                    f"  Step {step:4d} | Action: {ACTION_NAMES[action_int]:20s} | "
                    f"Reward: {reward:+.1f} | Total: {total_reward:.1f} | "
                    f"Top probs: [{top3_str}]"
                )

            # Check for new achievements
            for name, unlocked_flag in info.get("achievements", {}).items():
                if unlocked_flag and name not in unlocked:
                    unlocked.add(name)
                    print(f"  🏆 Achievement unlocked: {name}")

            if args.delay > 0:
                time.sleep(args.delay)

            if is_recurrent:
                done_tensor = torch.tensor([float(done)], device=device)

        print(f"\n  Episode {episode+1} complete:")
        print(f"  Total reward:  {total_reward:.2f}")
        print(f"  Steps:         {step}")
        print(f"  Achievements:  {len(unlocked)} — {', '.join(sorted(unlocked)) if unlocked else 'none'}")
        print(f"  Terminated:    {terminated}")


if __name__ == "__main__":
    main()

