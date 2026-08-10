#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate a trained checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt
    python scripts/evaluate.py --checkpoint checkpoints/crafter_lstm_ppo_latest.pt --episodes 50
"""

import sys
import os
import argparse
import logging
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch

from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
from crafter_ai.evaluation.evaluator import Evaluator
from crafter_ai.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_type = checkpoint.get("model_type", "ActorCritic")
    config = checkpoint.get("config", {})

    logger.info(f"Model type: {model_type}")
    logger.info(f"Checkpoint step: {checkpoint.get('global_step', 0)}")
    logger.info(f"Checkpoint episodes: {checkpoint.get('episode_count', 0)}")

    # Build model
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

    # Evaluate
    evaluator = Evaluator(
        model=model, device=device,
        num_episodes=args.episodes, seed=args.seed,
    )
    results = evaluator.evaluate(verbose=True)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({args.episodes} episodes)")
    print("=" * 60)
    print(f"Mean reward:    {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean length:    {results['mean_length']:.0f}")
    print(f"Crafter score:  {results['crafter_score']*100:.2f}%")
    print("\nAchievement Success Rates:")
    for name, rate in sorted(results["achievement_rates"].items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"  {name:30s} {bar} {rate*100:5.1f}%")

    # Save output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

