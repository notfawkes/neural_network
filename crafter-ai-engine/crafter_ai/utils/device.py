"""
Device utilities for Apple Silicon M4 (MPS) and CPU.
"""

import torch
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Detect and return the best available PyTorch device.

    Priority on Apple Silicon: MPS > CPU
    (CUDA is not available on M4)

    Parameters
    ----------
    prefer_mps : bool
        If True, prefer MPS over CPU when available.

    Returns
    -------
    torch.device
    """
    if prefer_mps and torch.backends.mps.is_available():
        try:
            # Quick MPS sanity check
            test = torch.zeros(1, device="mps")
            _ = test + 1
            logger.info("Using MPS device (Apple Silicon)")
            return torch.device("mps")
        except Exception as e:
            logger.warning(f"MPS available but test failed: {e}. Falling back to CPU.")

    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device("cuda")

    logger.info("Using CPU device")
    return torch.device("cpu")


def benchmark_device(device: torch.device, batch_size: int = 16) -> dict:
    """
    Benchmark CNN + LSTM forward pass speed on the given device.

    Returns timing information to help choose optimal settings.
    """
    from ..models.actor_critic import RecurrentActorCritic

    logger.info(f"Benchmarking device: {device}")
    model = RecurrentActorCritic().to(device)
    model.eval()

    obs = torch.randn(batch_size, 3, 64, 64).to(device)
    done = torch.zeros(batch_size).to(device)
    state = model.initial_state(batch_size, device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model.get_action_and_value(obs, done, state)

    # Benchmark
    n_trials = 50
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            model.get_action_and_value(obs, done, state)
    elapsed = time.perf_counter() - t0

    fps_estimate = (batch_size * n_trials) / elapsed

    result = {
        "device": str(device),
        "batch_size": batch_size,
        "n_trials": n_trials,
        "total_time_s": elapsed,
        "time_per_step_ms": (elapsed / n_trials) * 1000,
        "steps_per_second_estimate": fps_estimate,
    }

    logger.info(
        f"Benchmark [{device}]: {fps_estimate:.0f} steps/sec "
        f"({result['time_per_step_ms']:.2f}ms per step)"
    )

    return result

