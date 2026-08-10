"""
CrafterEnv Adapter
==================
Normalizes Crafter's custom API (which uses a minimal gym-like interface without
a full gym dependency) into a consistent interface that our training code expects.

The Crafter environment (v1.8.3) exposes:
  - obs = env.reset()           → (64, 64, 3) uint8
  - obs, reward, done, info = env.step(action)
  - env.action_space.n          → 17
  - env.action_names            → list of 17 strings
  - info["achievements"]        → dict of achievement_name: bool
  - info["player_pos"]          → [x, y] array
  - info["inventory"]           → dict
  - info["discount"]            → float
  - info["reward"]              → float
  - info["semantic"]            → semantic map array

Gym is NOT installed in the venv; we work directly with crafter.Env.

This adapter:
  1. Preprocesses observations: HWC → CHW, uint8 → float32 normalized [0,1]
  2. Exposes a stable reset() that returns (obs, info)
  3. Exposes a stable step() that returns (obs, reward, terminated, truncated, info)
  4. Tracks episode stats internally
  5. Provides player position from info (for visualization only, never to the NN)
"""

import sys
import os
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Ensure the crafter package is importable
_POSSIBLE_ROOTS = [
    "/Users/bala/crafter/crafter-game",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "crafter-game")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "crafter-game")),
    "/Users/bala/crafter",
]
for _root in _POSSIBLE_ROOTS:
    if os.path.isdir(os.path.join(_root, "crafter")) and _root not in sys.path:
        sys.path.insert(0, _root)
        break

import crafter as _crafter

# ─────────────────────────────────────────────────────────────────────────────
# Action names (17 actions)
# ─────────────────────────────────────────────────────────────────────────────
ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
]
NUM_ACTIONS = 17

# 22 achievements
ACHIEVEMENT_NAMES = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
]
NUM_ACHIEVEMENTS = 22

# Observation shape constants
OBS_H, OBS_W, OBS_C = 64, 64, 3
OBS_SHAPE_HWC = (OBS_H, OBS_W, OBS_C)
OBS_SHAPE_CHW = (OBS_C, OBS_H, OBS_W)  # PyTorch format


class CrafterEnv:
    """
    Single Crafter environment with normalized observations and
    a consistent Gymnasium-style step API.

    Observations returned are float32 tensors in CHW format, normalized [0, 1].
    Raw uint8 HWC observations are stored as self.last_raw_obs for visualization.

    Parameters
    ----------
    reward : bool
        If True, uses the shaped reward (+1 achievement, ±0.1 health).
        If False, no rewards (unsupervised mode).
    seed : int, optional
        Random seed for reproducibility.
    max_episode_steps : int
        Maximum steps per episode (Crafter default is 10000).
    record_raw : bool
        Keep raw uint8 observations for visualization/recording.
    """

    def __init__(
        self,
        reward: bool = True,
        seed: Optional[int] = None,
        max_episode_steps: int = 10000,
        record_raw: bool = True,
    ):
        self._env = _crafter.Env(reward=reward, length=max_episode_steps, seed=seed)
        self._reward_mode = reward
        self._max_episode_steps = max_episode_steps
        self._seed = seed
        self._record_raw = record_raw

        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_id = 0
        self._episode_start_time = 0.0
        self._total_steps = 0

        # Last raw observation (HWC uint8) for visualization
        self.last_raw_obs: Optional[np.ndarray] = None
        # Last processed observation (CHW float32) for the network
        self.last_obs: Optional[np.ndarray] = None
        # Last info dict
        self.last_info: Dict[str, Any] = {}
        # Last player position
        self.last_pos: Optional[np.ndarray] = None

        # Achievement tracking across episodes
        self._episode_achievements: Dict[str, bool] = {k: False for k in ACHIEVEMENT_NAMES}

        # Trajectory for visualization (player positions within episode)
        self._trajectory: List[Tuple[int, int]] = []

    # ─────────────────────────────────────────────────────────────────────
    # Spaces
    # ─────────────────────────────────────────────────────────────────────
    @property
    def observation_space_shape(self) -> Tuple[int, int, int]:
        """Shape of preprocessed observations: (C, H, W)."""
        return OBS_SHAPE_CHW

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def action_names(self) -> List[str]:
        return ACTION_NAMES

    # ─────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Returns
        -------
        obs : np.ndarray
            Preprocessed observation (3, 64, 64) float32 in [0, 1].
        info : dict
            Initial info dict.
        """
        if seed is not None:
            self._env._seed = seed

        raw_obs = self._env.reset()

        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_id += 1
        self._episode_start_time = time.time()
        self._episode_achievements = {k: False for k in ACHIEVEMENT_NAMES}
        self._trajectory = []

        if self._record_raw:
            self.last_raw_obs = raw_obs.copy()

        obs = self._preprocess(raw_obs)
        self.last_obs = obs

        # Build initial info
        info = {
            "episode_id": self._episode_id,
            "episode_step": 0,
            "episode_reward": 0.0,
            "player_pos": np.array([32, 32]),  # center of map
            "achievements": {k: False for k in ACHIEVEMENT_NAMES},
            "inventory": {},
            "total_steps": self._total_steps,
        }
        self.last_info = info
        self.last_pos = info["player_pos"]
        self._trajectory.append(tuple(info["player_pos"].tolist()))

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.

        Parameters
        ----------
        action : int
            Action index in [0, 16].

        Returns
        -------
        obs : np.ndarray
            Preprocessed observation (3, 64, 64) float32 in [0, 1].
        reward : float
            Scalar reward.
        terminated : bool
            True if the episode ended due to agent death / game over.
        truncated : bool
            True if the episode ended due to max step limit.
        info : dict
            Step metadata.
        """
        raw_obs, reward, done, raw_info = self._env.step(int(action))

        self._episode_step += 1
        self._total_steps += 1
        self._episode_reward += reward

        if self._record_raw:
            self.last_raw_obs = raw_obs.copy()

        obs = self._preprocess(raw_obs)
        self.last_obs = obs

        # Parse info
        env_achievements = raw_info.get("achievements", {})
        for k in ACHIEVEMENT_NAMES:
            if env_achievements.get(k, False):
                self._episode_achievements[k] = True

        player_pos = raw_info.get("player_pos", np.array([32, 32]))
        self.last_pos = player_pos
        self._trajectory.append(tuple(player_pos.tolist()))

        # Determine termination type
        # Crafter uses `done` for both max steps and death; we can distinguish
        # via discount field (discount=0 means terminal/death, 1 means truncated)
        discount = raw_info.get("discount", 0 if done else 1)
        terminated = done and (discount == 0)
        truncated = done and (discount != 0)

        info = {
            "episode_id": self._episode_id,
            "episode_step": self._episode_step,
            "episode_reward": self._episode_reward,
            "player_pos": player_pos,
            "achievements": dict(self._episode_achievements),
            "inventory": raw_info.get("inventory", {}),
            "total_steps": self._total_steps,
            "discount": discount,
            "raw_reward": raw_info.get("reward", reward),
            "semantic": raw_info.get("semantic", None),
            "trajectory": list(self._trajectory) if done else None,
        }
        self.last_info = info

        return obs, float(reward), terminated, truncated, info

    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Return current episode trajectory as list of (x, y) tuples."""
        return list(self._trajectory)

    # ─────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(obs: np.ndarray) -> np.ndarray:
        """Convert HWC uint8 [0,255] → CHW float32 [0,1]."""
        # obs: (64, 64, 3) uint8
        obs = obs.astype(np.float32) / 255.0
        # HWC → CHW
        obs = np.transpose(obs, (2, 0, 1))
        return obs

    @staticmethod
    def preprocess_batch(obs: np.ndarray) -> np.ndarray:
        """Preprocess a batch of HWC uint8 observations → NCHW float32."""
        # obs: (N, 64, 64, 3) uint8
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized Environment
# ─────────────────────────────────────────────────────────────────────────────

class VecCrafterEnv:
    """
    Synchronous vectorized Crafter environment.

    Runs N independent Crafter environments. Each has its own episode state.
    Autoreset: when an environment episode ends, it is automatically reset
    and the reset observation is returned.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    reward : bool
        Enable Crafter reward.
    seeds : list of int, optional
        Seeds for each env. If None, uses random seeds.
    max_episode_steps : int
        Max steps per episode.
    record_raw : bool
        Keep raw observations for recording.
    """

    def __init__(
        self,
        num_envs: int = 4,
        reward: bool = True,
        seeds: Optional[List[int]] = None,
        max_episode_steps: int = 10000,
        record_raw: bool = True,
    ):
        self.num_envs = num_envs
        self._max_episode_steps = max_episode_steps

        if seeds is None:
            rng = np.random.RandomState(42)
            seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(num_envs)]

        self.envs: List[CrafterEnv] = [
            CrafterEnv(
                reward=reward,
                seed=seeds[i],
                max_episode_steps=max_episode_steps,
                record_raw=record_raw,
            )
            for i in range(num_envs)
        ]

        # Track last obs for each env
        self._last_obs: Optional[np.ndarray] = None
        self._last_infos: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

        # For autoreset: store final obs before reset
        self._final_obs: List[Optional[np.ndarray]] = [None] * num_envs
        self._final_infos: List[Optional[Dict]] = [None] * num_envs

    @property
    def observation_space_shape(self) -> Tuple[int, int, int]:
        return OBS_SHAPE_CHW

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def action_names(self) -> List[str]:
        return ACTION_NAMES

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Reset all environments.

        Returns
        -------
        obs : np.ndarray
            Shape (num_envs, 3, 64, 64) float32.
        infos : list of dict
        """
        obs_list = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)
        self._last_obs = np.stack(obs_list, axis=0)
        self._last_infos = infos
        return self._last_obs, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with given actions. Auto-reset done envs.

        Parameters
        ----------
        actions : np.ndarray
            Shape (num_envs,) int.

        Returns
        -------
        obs : (num_envs, 3, 64, 64) float32
        rewards : (num_envs,) float32
        terminated : (num_envs,) bool
        truncated : (num_envs,) bool
        infos : list of dict
        """
        obs_list, rewards, terminated_arr, truncated_arr = [], [], [], []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            if done:
                # Store final observation before reset
                self._final_obs[i] = obs
                self._final_infos[i] = dict(info)
                info["final_obs"] = obs
                info["episode_done"] = True

                # Auto-reset
                obs, reset_info = env.reset()
                info["reset_info"] = reset_info
            else:
                self._final_obs[i] = None
                self._final_infos[i] = None
                info["episode_done"] = False

            obs_list.append(obs)
            rewards.append(reward)
            terminated_arr.append(terminated)
            truncated_arr.append(truncated)
            infos.append(info)

        self._last_obs = np.stack(obs_list, axis=0)
        self._last_infos = infos

        return (
            self._last_obs,
            np.array(rewards, dtype=np.float32),
            np.array(terminated_arr, dtype=bool),
            np.array(truncated_arr, dtype=bool),
            infos,
        )

    def get_raw_obs(self) -> List[Optional[np.ndarray]]:
        """Return raw uint8 HWC observations for each env."""
        return [env.last_raw_obs for env in self.envs]

    def get_positions(self) -> List[Optional[np.ndarray]]:
        """Return current player positions for visualization."""
        return [env.last_pos for env in self.envs]

    def close(self):
        """Close all environments."""
        # Crafter Env doesn't have close(), but we clear references
        self.envs = []


def make_env(
    num_envs: int = 4,
    reward: bool = True,
    seed: int = 42,
    max_episode_steps: int = 10000,
) -> VecCrafterEnv:
    """Factory function to create a VecCrafterEnv."""
    seeds = [seed + i for i in range(num_envs)]
    return VecCrafterEnv(
        num_envs=num_envs,
        reward=reward,
        seeds=seeds,
        max_episode_steps=max_episode_steps,
        record_raw=True,
    )
