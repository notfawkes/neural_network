#!/usr/bin/env python3
"""
train.py
========
Launch PPO training (MLP baseline or CNN + LSTM recurrent).

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/default.yaml --resume checkpoints/crafter_lstm_ppo_latest.pt
"""

import sys
import os
import argparse
import logging
import yaml
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import numpy as np

from crafter_ai.env.adapter import make_env, VecCrafterEnv
from crafter_ai.models.actor_critic import ActorCritic, RecurrentActorCritic
from crafter_ai.training.ppo import PPOTrainer
from crafter_ai.training.ppo_recurrent import RecurrentPPOTrainer
from crafter_ai.recording.episode_recorder import EpisodeRecorder
from crafter_ai.recording.storage import RecordingStorage
from crafter_ai.utils.device import get_device
from crafter_ai.utils.metrics import MetricsTracker


def setup_logging(log_dir: str, experiment_name: str):
    log_path = Path(log_dir) / f"{experiment_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path)),
        ],
    )
    return logging.getLogger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict, device: torch.device):
    model_type = config["experiment"]["model_type"]
    mc = config.get("model", {})
    tc = config.get("training", {})

    if model_type == "RecurrentActorCritic":
        model = RecurrentActorCritic(
            in_channels=mc.get("in_channels", 3),
            num_actions=mc.get("num_actions", 17),
            cnn_out=mc.get("cnn_out_features", 512),
            lstm_hidden=mc.get("lstm_hidden_size", 256) or tc.get("lstm_hidden_size", 256),
            lstm_layers=mc.get("lstm_layers", 1),
        ).to(device)
    else:
        model = ActorCritic(
            in_channels=mc.get("in_channels", 3),
            num_actions=mc.get("num_actions", 17),
            cnn_out=mc.get("cnn_out_features", 512),
        ).to(device)

    return model, model_type


def main():
    parser = argparse.ArgumentParser(description="Train CRAfter AI agent")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]
    tc = config["training"]
    rc = config.get("recording", {})
    lc = config.get("logging", {})

    # Merge total_timesteps override
    if args.total_timesteps:
        tc["total_timesteps"] = args.total_timesteps

    logger = setup_logging(lc.get("log_dir", "logs"), exp["name"])
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {exp['name']}")

    # Seed
    seed = exp.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    dc = config.get("device", {})
    device = get_device(prefer_mps=dc.get("prefer_mps", True))
    logger.info(f"Device: {device}")

    # Build environments
    num_envs = tc.get("num_envs", 4)
    env = make_env(
        num_envs=num_envs,
        reward=config.get("environment", {}).get("reward", True),
        seed=seed,
        max_episode_steps=config.get("environment", {}).get("max_episode_steps", 10000),
    )
    logger.info(f"Environments created: {num_envs}")

    # Build model
    model, model_type = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_type}, parameters: {total_params:,}")

    # Merge training config
    flat_tc = dict(tc)
    flat_tc.update(config.get("model", {}))
    flat_tc["experiment_name"] = exp["name"]

    # Build trainer
    cc = config.get("checkpointing", {})
    if model_type == "RecurrentActorCritic":
        trainer = RecurrentPPOTrainer(
            model=model,
            env=env,
            config=flat_tc,
            device=device,
            checkpoint_dir=cc.get("checkpoint_dir", "checkpoints"),
            experiment_name=exp["name"],
        )
    else:
        trainer = PPOTrainer(
            model=model,
            env=env,
            config=flat_tc,
            device=device,
            checkpoint_dir=cc.get("checkpoint_dir", "checkpoints"),
            experiment_name=exp["name"],
        )

    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Set up recording
    recorder = EpisodeRecorder(
        storage_dir=rc.get("storage_dir", "recordings"),
        enabled=rc.get("enabled", True),
        frame_interval=rc.get("frame_interval", 2),
        compression=rc.get("compression", True),
        max_storage_gb=rc.get("max_storage_gb", 5.0),
        experiment_name=exp["name"],
    )

    # Metrics tracker
    metrics = MetricsTracker(
        log_dir=lc.get("log_dir", "logs"),
        experiment_name=exp["name"],
    )

    # Shared state storage for backend
    storage = RecordingStorage(
        base_dir="experiments",
        experiment_name=exp["name"],
    )

    # Recording state
    _active_recording = {}  # env_idx → recording active

    def metrics_callback(event: dict):
        event_type = event.get("type")

        if event_type == "episode_done":
            ep_id = event.get("episode_id", 0)
            reward = event.get("reward", 0.0)
            length = event.get("length", 0)
            achievements = event.get("achievements", {})

            metrics.record_episode(
                reward=reward,
                length=length,
                global_step=event.get("step", 0),
                achievements=achievements,
            )

            # Update shared state
            stats = metrics.get_episode_stats()
            storage.update_training_state({
                "status": "training",
                "global_step": trainer.global_step,
                "episode_count": trainer.episode_count,
                "mean_reward": stats.get("mean_reward", 0.0),
                "mean_ep_length": stats.get("mean_length", 0.0),
                "crafter_score": metrics.get_crafter_score(),
                "fps": event.get("fps", 0.0),
                "last_episode_reward": reward,
                "last_episode_length": length,
            })

        elif event_type == "update":
            metrics.record_update(event)
            storage.update_training_state({
                "status": "training",
                "global_step": trainer.global_step,
                "update_count": trainer.update_count,
                "policy_loss": event.get("policy_loss", 0.0),
                "value_loss": event.get("value_loss", 0.0),
                "entropy": abs(event.get("entropy_loss", 0.0)),
                "approx_kl": event.get("approx_kl", 0.0),
                "fps": event.get("fps", 0.0),
            })

            # Save TensorBoard metrics
            if event.get("update_count", 0) % 10 == 0:
                metrics.save()

    # Update storage status
    storage.update_training_state({"status": "starting", "experiment_name": exp["name"]})

    logger.info("Starting training...")
    try:
        trainer.train(
            total_timesteps=tc.get("total_timesteps", 1_000_000),
            metrics_callback=metrics_callback,
            checkpoint_interval=cc.get("checkpoint_interval", 50_000),
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        trainer.save_checkpoint(tag="interrupted")
    finally:
        storage.update_training_state({"status": "stopped"})
        metrics.save()
        recorder.stop()

    logger.info("Training finished.")


if __name__ == "__main__":
    main()

