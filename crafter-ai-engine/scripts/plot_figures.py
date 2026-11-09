#!/usr/bin/env python3
"""
plot_figures.py
===============
Generate all types of training and evaluation figures for Crafter AI.

Usage:
    python scripts/plot_figures.py
    python scripts/plot_figures.py --output-dir figures --experiment crafter_lstm_ppo
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from crafter_ai.evaluation.figures import FigureGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_figures")


def main():
    parser = argparse.ArgumentParser(description="Generate all training and evaluation figures")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save figures")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory containing training logs")
    parser.add_argument("--recordings-dir", type=str, default=None, help="Directory containing recorded episodes")
    parser.add_argument("--checkpoints-dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--experiment", type=str, default="crafter_lstm_ppo", help="Experiment name")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parent.parent.parent

    output_dir = Path(args.output_dir) if args.output_dir else workspace_root / "figures"
    log_dir = Path(args.log_dir) if args.log_dir else workspace_root / "logs"
    recordings_dir = Path(args.recordings_dir) if args.recordings_dir else workspace_root / "recordings"
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else workspace_root / "checkpoints"

    logger.info(f"Target figures directory: {output_dir}")
    logger.info(f"Source logs directory: {log_dir}")
    logger.info(f"Source recordings directory: {recordings_dir}")
    logger.info(f"Source checkpoints directory: {checkpoints_dir}")

    fig_gen = FigureGenerator(
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        recordings_dir=str(recordings_dir),
        checkpoints_dir=str(checkpoints_dir),
    )

    generated = fig_gen.generate_all_figures(experiment_name=args.experiment)
    logger.info(f"Complete! Generated {len(generated)} publication figures:")
    for p in generated:
        logger.info(f"  ✓ {p.name}")


if __name__ == "__main__":
    main()
