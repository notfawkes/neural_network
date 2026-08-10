"""
Episode Recorder
================

Records complete episode data to disk using a background writer thread.
Training is never blocked by I/O — recordings are queued and written asynchronously.

Recorded data per episode:
  - obs_frames: list of uint8 HWC images (compressed with zlib)
  - actions: list of int
  - rewards: list of float
  - dones: bool
  - action_probs: list of float arrays
  - critic_values: list of float
  - infos: metadata
  - global_step: when recorded
  - episode_id: int

Storage format: numpy .npz files (compressed)
"""

import os
import io
import time
import queue
import threading
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from ..env.adapter import ACHIEVEMENT_NAMES

logger = logging.getLogger(__name__)


@dataclass
class EpisodeData:
    """Container for complete episode data."""
    episode_id: int
    global_step: int
    start_time: float = 0.0
    end_time: float = 0.0

    # Observations (raw HWC uint8)
    obs_frames: List[np.ndarray] = field(default_factory=list)

    # Actions, rewards, done flags
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False

    # Model outputs (from agent)
    action_probs: List[Optional[List[float]]] = field(default_factory=list)
    critic_values: List[Optional[float]] = field(default_factory=list)
    lstm_h_norms: List[Optional[float]] = field(default_factory=list)

    # Player positions for trajectory visualization
    player_positions: List[List[int]] = field(default_factory=list)

    # Achievements unlocked
    achievements: Dict[str, bool] = field(default_factory=dict)

    # Metadata
    experiment_name: str = ""
    model_type: str = ""

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_achievements(self) -> int:
        return sum(1 for v in self.achievements.values() if v)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict (without large obs arrays)."""
        return {
            "episode_id": self.episode_id,
            "global_step": self.global_step,
            "total_reward": self.total_reward,
            "length": self.length,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "achievements": self.achievements,
            "num_achievements": self.num_achievements,
            "duration_seconds": self.duration_seconds,
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "player_positions": self.player_positions,
        }


class EpisodeRecorder:
    """
    Records episode data and writes to disk asynchronously.

    Uses a background thread with a bounded queue so training is never blocked.
    If the queue is full, the episode is dropped (with a warning).

    Parameters
    ----------
    storage_dir : str
        Directory to save episode recordings.
    enabled : bool
        If False, all recording is disabled.
    frame_interval : int
        Record every Nth frame (1 = all frames).
    compression : bool
        Compress saved arrays.
    max_storage_gb : float
        Maximum storage to use. When exceeded, old recordings are pruned.
    queue_size : int
        Maximum number of episodes in the write queue.
    """

    def __init__(
        self,
        storage_dir: str = "recordings",
        enabled: bool = True,
        frame_interval: int = 1,
        compression: bool = True,
        max_storage_gb: float = 5.0,
        queue_size: int = 20,
        experiment_name: str = "default",
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.frame_interval = frame_interval
        self.compression = compression
        self.max_storage_bytes = int(max_storage_gb * 1024 ** 3)
        self.experiment_name = experiment_name

        # Current episode being recorded
        self._current: Optional[EpisodeData] = None
        self._frame_count = 0

        # Background writer
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="EpisodeWriter"
        )
        self._stop_event = threading.Event()
        if enabled:
            self._writer_thread.start()

        # Episode index (episode_id → metadata dict)
        self._episode_index: Dict[int, Dict[str, Any]] = {}
        self._load_index()

    def begin_episode(
        self,
        episode_id: int,
        global_step: int,
        experiment_name: str = "",
        model_type: str = "",
    ):
        """Start recording a new episode."""
        if not self.enabled:
            return
        self._current = EpisodeData(
            episode_id=episode_id,
            global_step=global_step,
            start_time=time.time(),
            experiment_name=experiment_name,
            model_type=model_type,
        )
        self._frame_count = 0

    def record_step(
        self,
        obs: Optional[np.ndarray],
        action: int,
        reward: float,
        player_pos: Optional[List[int]] = None,
        action_probs: Optional[np.ndarray] = None,
        critic_value: Optional[float] = None,
        lstm_h_norm: Optional[float] = None,
    ):
        """Record a single environment step."""
        if not self.enabled or self._current is None:
            return

        self._frame_count += 1

        if obs is not None and (self._frame_count - 1) % self.frame_interval == 0:
            self._current.obs_frames.append(obs.copy() if isinstance(obs, np.ndarray) else obs)

        self._current.actions.append(int(action))
        self._current.rewards.append(float(reward))

        if player_pos is not None:
            self._current.player_positions.append(
                player_pos.tolist() if isinstance(player_pos, np.ndarray) else player_pos
            )

        if action_probs is not None:
            self._current.action_probs.append(
                action_probs.tolist() if isinstance(action_probs, np.ndarray) else list(action_probs)
            )
        else:
            self._current.action_probs.append(None)

        self._current.critic_values.append(
            float(critic_value) if critic_value is not None else None
        )
        self._current.lstm_h_norms.append(
            float(lstm_h_norm) if lstm_h_norm is not None else None
        )

    def end_episode(
        self,
        achievements: Optional[Dict[str, bool]] = None,
        terminated: bool = False,
        truncated: bool = False,
    ):
        """Finalize and queue the episode for writing."""
        if not self.enabled or self._current is None:
            return

        self._current.end_time = time.time()
        self._current.terminated = terminated
        self._current.truncated = truncated
        self._current.achievements = achievements or {k: False for k in ACHIEVEMENT_NAMES}

        # Queue for async write
        try:
            self._queue.put_nowait(self._current)
        except queue.Full:
            logger.warning(
                f"Recording queue full. Dropping episode {self._current.episode_id}. "
                "Consider increasing queue_size or reducing frame_interval."
            )

        self._current = None

    def _writer_loop(self):
        """Background thread: write episodes from queue to disk."""
        while not self._stop_event.is_set():
            try:
                episode = self._queue.get(timeout=1.0)
                self._write_episode(episode)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Episode writer error: {e}", exc_info=True)
                # Never crash training due to recording errors

    def _write_episode(self, episode: EpisodeData):
        """Write episode to disk as compressed npz."""
        try:
            path = self.storage_dir / f"ep_{episode.episode_id:08d}.npz"

            # Build arrays
            arrays = {
                "actions": np.array(episode.actions, dtype=np.int16),
                "rewards": np.array(episode.rewards, dtype=np.float32),
                "player_positions": np.array(episode.player_positions, dtype=np.int16),
            }

            if episode.obs_frames:
                arrays["obs_frames"] = np.stack(episode.obs_frames, axis=0)

            if any(p is not None for p in episode.action_probs):
                probs = [p if p is not None else [0.0] * 17 for p in episode.action_probs]
                arrays["action_probs"] = np.array(probs, dtype=np.float32)

            if any(v is not None for v in episode.critic_values):
                vals = [v if v is not None else 0.0 for v in episode.critic_values]
                arrays["critic_values"] = np.array(vals, dtype=np.float32)

            # Save
            if self.compression:
                np.savez_compressed(str(path), **arrays)
            else:
                np.savez(str(path), **arrays)

            # Update index
            meta = episode.to_dict()
            meta["path"] = str(path)
            self._episode_index[episode.episode_id] = meta
            self._save_index()

            # Check storage limit
            self._check_storage_limit()

        except Exception as e:
            logger.error(f"Failed to write episode {episode.episode_id}: {e}", exc_info=True)

    def _load_index(self):
        """Load episode index from disk."""
        index_path = self.storage_dir / f"{self.experiment_name}_index.json"
        if index_path.exists():
            import json
            with open(index_path) as f:
                data = json.load(f)
            self._episode_index = {int(k): v for k, v in data.items()}

    def _save_index(self):
        """Save episode index to disk."""
        import json
        index_path = self.storage_dir / f"{self.experiment_name}_index.json"
        with open(index_path, "w") as f:
            json.dump({str(k): v for k, v in self._episode_index.items()}, f)

    def _check_storage_limit(self):
        """Prune oldest recordings if storage limit exceeded."""
        total_size = sum(
            os.path.getsize(v["path"])
            for v in self._episode_index.values()
            if os.path.exists(v.get("path", ""))
        )

        if total_size > self.max_storage_bytes:
            # Remove oldest episodes
            sorted_ids = sorted(self._episode_index.keys())
            while total_size > self.max_storage_bytes * 0.9 and sorted_ids:
                oldest_id = sorted_ids.pop(0)
                entry = self._episode_index.get(oldest_id)
                if entry and os.path.exists(entry.get("path", "")):
                    size = os.path.getsize(entry["path"])
                    os.unlink(entry["path"])
                    total_size -= size
                    del self._episode_index[oldest_id]

            self._save_index()
            logger.info(f"Pruned old recordings. Current usage: {total_size/1e9:.2f} GB")

    def get_episode_metadata(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific episode."""
        return self._episode_index.get(episode_id)

    def list_episodes(self) -> List[Dict[str, Any]]:
        """List all recorded episodes sorted by episode_id."""
        return sorted(self._episode_index.values(), key=lambda x: x["episode_id"])

    def load_episode_frames(self, episode_id: int) -> Optional[np.ndarray]:
        """Load obs frames for an episode. Returns (T, 64, 64, 3) uint8 or None."""
        meta = self._episode_index.get(episode_id)
        if meta is None or not os.path.exists(meta.get("path", "")):
            return None
        try:
            data = np.load(meta["path"])
            return data.get("obs_frames")
        except Exception as e:
            logger.error(f"Failed to load frames for episode {episode_id}: {e}")
            return None

    def load_episode_full(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Load full episode data including obs frames."""
        meta = self._episode_index.get(episode_id)
        if meta is None or not os.path.exists(meta.get("path", "")):
            return None
        try:
            data = np.load(meta["path"], allow_pickle=True)
            result = dict(meta)
            for key in data.files:
                result[key] = data[key]
            return result
        except Exception as e:
            logger.error(f"Failed to load episode {episode_id}: {e}")
            return None

    def stop(self):
        """Stop the background writer thread gracefully."""
        self._stop_event.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

