"""
Environment wrappers for additional functionality.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from collections import deque
from .adapter import CrafterEnv


class FrameStack(CrafterEnv):
    """
    Stack the last N frames along the channel dimension.

    With n_stack=4: obs shape becomes (12, 64, 64) instead of (3, 64, 64).
    """

    def __init__(self, n_stack: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_stack = n_stack
        self._frames: deque = deque(maxlen=n_stack)
        # Update observation shape
        self._stacked_shape = (3 * n_stack, 64, 64)

    @property
    def observation_space_shape(self) -> Tuple[int, int, int]:
        return self._stacked_shape

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed)
        # Fill buffer with copies of the first frame
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._get_stacked(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self._frames.append(obs)
        return self._get_stacked(), reward, terminated, truncated, info

    def _get_stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)


class RecordingWrapper(CrafterEnv):
    """
    Wraps CrafterEnv and records complete episode data.

    Recorded data per episode:
      - raw_obs: list of uint8 HWC images
      - actions: list of int
      - rewards: list of float
      - terminated: list of bool
      - truncated: list of bool
      - action_probs: list of float arrays (set externally)
      - critic_values: list of float (set externally)
      - infos: list of dict
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._recording_active = False
        self._current_episode_data: Optional[Dict[str, Any]] = None
        self._completed_episodes: List[Dict[str, Any]] = []

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(seed=seed)

        self._recording_active = True
        self._current_episode_data = {
            "episode_id": self._episode_id,
            "start_time": info.get("total_steps", 0),
            "raw_obs": [self.last_raw_obs.copy() if self.last_raw_obs is not None else None],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "action_probs": [],
            "critic_values": [],
            "infos": [info],
            "player_positions": [info.get("player_pos", np.array([32, 32])).tolist()],
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        if self._recording_active and self._current_episode_data is not None:
            self._current_episode_data["actions"].append(int(action))
            self._current_episode_data["rewards"].append(float(reward))
            self._current_episode_data["terminated"].append(bool(terminated))
            self._current_episode_data["truncated"].append(bool(truncated))
            self._current_episode_data["action_probs"].append(None)  # filled externally
            self._current_episode_data["critic_values"].append(None)  # filled externally
            self._current_episode_data["infos"].append(info)
            pos = info.get("player_pos", np.array([32, 32]))
            self._current_episode_data["player_positions"].append(pos.tolist())

            if self.last_raw_obs is not None:
                self._current_episode_data["raw_obs"].append(self.last_raw_obs.copy())

            if terminated or truncated:
                self._finalize_episode(info)

        return obs, reward, terminated, truncated, info

    def record_model_outputs(
        self,
        action_probs: Optional[np.ndarray],
        critic_value: Optional[float],
        step_idx: int = -1,
    ):
        """Called externally to record model outputs for the current step."""
        if self._current_episode_data is None:
            return
        probs = self._current_episode_data["action_probs"]
        values = self._current_episode_data["critic_values"]
        if step_idx == -1 and probs:
            probs[-1] = action_probs.tolist() if action_probs is not None else None
            values[-1] = float(critic_value) if critic_value is not None else None

    def _finalize_episode(self, final_info: Dict):
        """Save episode data to completed list."""
        if self._current_episode_data is None:
            return

        data = self._current_episode_data
        data["total_reward"] = final_info.get("episode_reward", sum(data["rewards"]))
        data["length"] = final_info.get("episode_step", len(data["actions"]))
        data["achievements"] = final_info.get("achievements", {})
        data["end_time"] = final_info.get("total_steps", 0)
        data["terminated"] = any(data["terminated"])
        data["truncated"] = data["truncated"][-1] if data["truncated"] else False

        self._completed_episodes.append(data)
        self._current_episode_data = None

    def get_completed_episodes(self) -> List[Dict[str, Any]]:
        return self._completed_episodes

    def pop_completed_episodes(self) -> List[Dict[str, Any]]:
        """Return and clear the completed episodes list."""
        eps = self._completed_episodes
        self._completed_episodes = []
        return eps

