"""
Recording Storage
=================
Manages the shared state between the training process and the backend server.
Uses a shared directory structure for inter-process communication.
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


class RecordingStorage:
    """
    Shared storage interface for live training state.

    The training process writes to this storage.
    The backend server reads from it.

    Uses JSON files + a simple file-based lock for coordination.

    Parameters
    ----------
    base_dir : str
        Base directory for all storage.
    experiment_name : str
    """

    def __init__(self, base_dir: str, experiment_name: str = "default"):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name

        # Subdirectories
        self.state_dir = self.base_dir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._state_path = self.state_dir / f"{experiment_name}_state.json"
        self._live_obs_path = self.state_dir / f"{experiment_name}_live_obs.npy"

        # In-memory state
        self._state: Dict[str, Any] = {
            "status": "idle",
            "global_step": 0,
            "episode_count": 0,
            "update_count": 0,
            "mean_reward": 0.0,
            "mean_ep_length": 0.0,
            "fps": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "crafter_score": 0.0,
            "last_update": time.time(),
            "experiment_name": experiment_name,
        }

    def update_training_state(self, state: Dict[str, Any]):
        """Update the training state (called from training process)."""
        with self._lock:
            self._state.update(state)
            self._state["last_update"] = time.time()
            self._flush_state()

    def _flush_state(self):
        """Write state to disk atomically."""
        import os
        try:
            tmp_path = self._state_path.with_suffix(".tmp.json")
            with open(tmp_path, "w") as f:
                json.dump(self._state, f, default=_json_default)
            os.replace(str(tmp_path), str(self._state_path))
        except Exception:
            pass  # Never block training

    def get_state(self) -> Dict[str, Any]:
        """Get current training state."""
        with self._lock:
            return dict(self._state)

    def update_live_obs(self, obs: "np.ndarray"):
        """Write latest observation for live streaming atomically."""
        import numpy as np
        import os
        try:
            tmp_path = self._live_obs_path.with_suffix(".tmp.npy")
            np.save(str(tmp_path), obs)
            os.replace(str(tmp_path), str(self._live_obs_path))
        except Exception:
            pass

    def get_live_obs(self) -> Optional["np.ndarray"]:
        """Read latest observation."""
        import numpy as np
        try:
            if self._live_obs_path.exists():
                return np.load(str(self._live_obs_path))
        except Exception:
            pass
        return None

