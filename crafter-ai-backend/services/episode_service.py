"""Episode service for listing, reading, and streaming episode replays."""
import io
import json
import base64
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class EpisodeService:
    def __init__(self, recordings_dir: Path):
        self.recordings_dir = recordings_dir

    def list_episodes(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all recorded episodes from index files or directory scan."""
        index_files = list(self.recordings_dir.glob("*_index.json"))
        episodes_map = {}

        for idx_file in index_files:
            try:
                with open(idx_file, "r") as f:
                    data = json.load(f)
                    for k, ep in data.items():
                        episodes_map[int(k)] = ep
            except Exception as e:
                logger.error(f"Error reading index {idx_file}: {e}")

        # If no index, scan .npz files directly
        if not episodes_map:
            for npz_file in sorted(self.recordings_dir.glob("ep_*.npz"), reverse=True):
                try:
                    # extract id from ep_00000001.npz
                    stem = npz_file.stem
                    ep_id = int(stem.split("_")[1])
                    episodes_map[ep_id] = {
                        "episode_id": ep_id,
                        "global_step": 0,
                        "total_reward": 0.0,
                        "length": 0,
                        "terminated": True,
                        "truncated": False,
                        "num_achievements": 0,
                        "achievements": {},
                        "duration_seconds": 0.0,
                        "experiment_name": "recorded",
                        "model_type": "RecurrentActorCritic",
                        "path": str(npz_file),
                    }
                except Exception:
                    continue

        sorted_eps = sorted(episodes_map.values(), key=lambda x: x["episode_id"], reverse=True)
        return sorted_eps[offset : offset + limit]

    def get_episode_detail(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get full metadata and trajectory for an episode."""
        ep_file = self.recordings_dir / f"ep_{episode_id:08d}.npz"
        if not ep_file.exists():
            # Search case-insensitively or check any match
            candidates = list(self.recordings_dir.glob(f"*_{episode_id:08d}.npz"))
            if candidates:
                ep_file = candidates[0]
            else:
                return None

        try:
            data = np.load(str(ep_file), allow_pickle=True)
            actions = data["actions"].tolist() if "actions" in data else []
            rewards = data["rewards"].tolist() if "rewards" in data else []
            positions = data["player_positions"].tolist() if "player_positions" in data else []
            critic_values = data["critic_values"].tolist() if "critic_values" in data else []
            action_probs = data["action_probs"].tolist() if "action_probs" in data else []

            # Read metadata from index if present
            meta = {
                "episode_id": episode_id,
                "global_step": 0,
                "total_reward": sum(rewards),
                "length": len(actions),
                "terminated": True,
                "truncated": False,
                "num_achievements": 0,
                "achievements": {},
                "duration_seconds": 0.0,
                "experiment_name": "default",
                "model_type": "RecurrentActorCritic",
                "actions": actions,
                "rewards": rewards,
                "player_positions": positions,
                "critic_values": critic_values,
                "action_probs": action_probs,
                "lstm_h_norms": [],
            }

            # Check index for additional stats
            for idx_file in self.recordings_dir.glob("*_index.json"):
                with open(idx_file) as f:
                    idx_data = json.load(f)
                    if str(episode_id) in idx_data:
                        meta.update(idx_data[str(episode_id)])
                        # preserve array fields
                        meta["actions"] = actions
                        meta["rewards"] = rewards
                        meta["player_positions"] = positions
                        meta["critic_values"] = critic_values
                        meta["action_probs"] = action_probs
                        break

            return meta
        except Exception as e:
            logger.error(f"Error loading episode {episode_id}: {e}")
            return None

    def get_episode_frames(self, episode_id: int, scale_factor: int = 1) -> Optional[List[str]]:
        """Return frames for an episode encoded as base64 JPEG strings."""
        ep_file = self.recordings_dir / f"ep_{episode_id:08d}.npz"
        if not ep_file.exists():
            candidates = list(self.recordings_dir.glob(f"*_{episode_id:08d}.npz"))
            if candidates:
                ep_file = candidates[0]
            else:
                return None

        try:
            data = np.load(str(ep_file), allow_pickle=True)
            if "obs_frames" not in data:
                return []
            
            raw_frames = data["obs_frames"]  # (T, 64, 64, 3) uint8
            encoded_frames = []

            for frame in raw_frames:
                img = Image.fromarray(frame)
                if scale_factor > 1:
                    img = img.resize(
                        (64 * scale_factor, 64 * scale_factor),
                        Image.NEAREST
                    )
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                encoded_frames.append(b64)

            return encoded_frames
        except Exception as e:
            logger.error(f"Error loading frames for episode {episode_id}: {e}")
            return None
